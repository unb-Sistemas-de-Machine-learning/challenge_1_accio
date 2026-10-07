import asyncio
import logging
import os
import secrets
from collections import OrderedDict

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, LinkPreviewOptions, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.helpers import escape_markdown

from fato_unb import observability
from fato_unb.bots.privacy import FiltroPII, scrub
from fato_unb.ingestion.scheduler import pipeline_job
from fato_unb.llm.checker import FactChecker
from fato_unb.llm.clients import LLMError, criar_cliente
from fato_unb.llm.guardrails import GuardrailConfig
from fato_unb.rag.embeddings import EmbeddingService
from fato_unb.rag.models import FonteCitada, VereditoJSON, VereditoType
from fato_unb.rag.pipeline import IndexingPipeline
from fato_unb.rag.reranker import Reranker
from fato_unb.rag.retriever import Retriever

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger().addFilter(FiltroPII())

logger = logging.getLogger(__name__)

TEXTO_AJUDA = (
    "Olá, sou o FatoUnB, o bot que te ajuda a identificar fakenews na UnB!\n\n"
    "Como usar em grupos:\n"
    "Copie a afirmação suspeita e mande junto do comando:\n"
    "/checar O RU vai fechar em outubro?\n\n"
    "Por privacidade (Privacy Mode do Telegram), eu não consigo ler sozinho o "
    "texto de uma mensagem de outra pessoa só respondendo a ela — por isso "
    "preciso que você cole o texto junto do comando.\n\n"
    "Só leio mensagens em que sou chamado com /checar. /privacidade explica o "
    "que faço com os dados."
)

TEXTO_PRIVACIDADE = (
    "Uso o texto que você me manda só para checar a afirmação na hora, sem "
    "guardar mensagens do grupo.\n\n"
    "Antes de qualquer análise, removo automaticamente CPFs, matrículas da "
    "UnB, e-mails e telefones que aparecerem no texto.\n\n"
    "Não processo nem registro mensagens em que eu não seja chamado "
    "diretamente com /checar."
)

TITULO_VEREDITO = {
    VereditoType.CONFIRMADO_OFICIALMENTE: ("✅", "Confirmado Oficialmente"),
    VereditoType.BOATO_SEM_REGISTRO: ("❌", "É Boato"),
    VereditoType.DESATUALIZADO_OU_FORA_DE_CONTEXTO: ("⚠️", "Desatualizado"),
    VereditoType.INCONCLUSIVO: ("❓", "Sem Confirmação"),
}

CONFIG_BOT = GuardrailConfig(max_chars_justificativa=280)
INTERVALO_INGESTAO_MIN = int(os.getenv("INGESTAO_INTERVALO_MIN", "60"))

_embedder: EmbeddingService | None = None
_fact_checker: FactChecker | None = None

_LIMITE_CACHE_FONTES = 200
_fontes_por_token: "OrderedDict[str, list[FonteCitada]]" = OrderedDict()


def obter_embedder() -> EmbeddingService:
    """Instância única de embedding, compartilhada entre o checar e a reindexação periódica.

    Criar uma por chamada carregaria o modelo (2+ GB) em dobro na memória.
    """
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingService(provider="local")
    return _embedder


def obter_fact_checker() -> FactChecker:
    global _fact_checker
    if _fact_checker is None:
        llm = criar_cliente()
        reranker_model = os.getenv("RERANKER_MODEL")
        retriever = Retriever(
            embedder=obter_embedder(),
            reranker=Reranker(model_name=reranker_model) if reranker_model else None,
        )
        _fact_checker = FactChecker(retriever, llm, config=CONFIG_BOT)
    return _fact_checker


async def ingestao_periodica(context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        pipeline = IndexingPipeline(embedder=obter_embedder())
        relatorio = await pipeline_job(pipeline=pipeline)
        logger.info("Ingestão periódica concluída: %s", relatorio)
    except Exception:
        logger.exception("Falha na ingestão periódica")


def guardar_fontes(fontes: list[FonteCitada]) -> str:
    token = secrets.token_hex(6)
    _fontes_por_token[token] = fontes
    while len(_fontes_por_token) > _LIMITE_CACHE_FONTES:
        _fontes_por_token.popitem(last=False)
    return token


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Olá! Eu sou o FatoUnB. Use /help para ver como me usar."
    )


async def ajuda(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(TEXTO_AJUDA.format(username=context.bot.username))


async def privacidade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(TEXTO_PRIVACIDADE)


def verificar_afirmacao(texto: str) -> VereditoJSON:
    try:
        checker = obter_fact_checker()
    except LLMError:
        logger.exception("Não consegui criar o cliente de LLM")
        return VereditoJSON(
            veredito=VereditoType.INCONCLUSIVO,
            justificativa=(
                "Não consegui acessar o modelo de linguagem agora (verifique a "
                "chave de API configurada). Tente novamente mais tarde."
            ),
            fontes=[],
            confianca=0.0,
            afirmacao_analisada=texto,
        )

    checagem = checker.verificar(texto)
    return checagem.veredito


def formatar_veredito(veredito: VereditoJSON) -> str:
    emoji, rotulo = TITULO_VEREDITO[veredito.veredito]

    linhas = [
        f"{emoji} *{escape_markdown(rotulo, version=2)}*",
        "",
        escape_markdown(veredito.justificativa, version=2),
        "",
        f"Confiança: {escape_markdown(f'{veredito.confianca:.0%}', version=2)}",
    ]

    return "\n".join(linhas)


def formatar_fontes(fontes: list[FonteCitada]) -> str:
    linhas = ["*Fontes:*"]
    for fonte in fontes:
        titulo = escape_markdown(fonte.title, version=2)
        url = escape_markdown(str(fonte.url), version=2, entity_type="text_link")
        linhas.append(f"• [{titulo}]({url})")
    return "\n".join(linhas)


def montar_teclado_fontes(veredito: VereditoJSON) -> InlineKeyboardMarkup | None:
    if not veredito.fontes:
        return None
    token = guardar_fontes(veredito.fontes)
    n = len(veredito.fontes)
    rotulo = f"📎 Ver fonte{'s' if n != 1 else ''} ({n})"
    return InlineKeyboardMarkup([[InlineKeyboardButton(rotulo, callback_data=f"fontes:{token}")]])


def extrair_afirmacao(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    resposta = update.message.reply_to_message
    if resposta:
        texto = resposta.text or resposta.caption
        if texto:
            return texto
    if context.args:
        return " ".join(context.args)
    return ""


def entidades_mencao_ao_bot(message, username: str) -> list:
    if not message.text or not message.entities:
        return []
    alvo = f"@{username}".lower()
    return [
        entidade
        for entidade in message.entities
        if entidade.type == "mention"
        and message.text[entidade.offset : entidade.offset + entidade.length].lower()
        == alvo
    ]


def remover_mencoes(texto: str, entidades: list) -> str:
    for entidade in sorted(entidades, key=lambda e: e.offset, reverse=True):
        inicio, fim = entidade.offset, entidade.offset + entidade.length
        texto = texto[:inicio] + texto[fim:]
    return texto.strip()


async def mencao(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None:
        return

    entidades = entidades_mencao_ao_bot(message, context.bot.username)
    if not entidades:
        return

    resposta = message.reply_to_message
    afirmacao = ""
    if resposta:
        afirmacao = resposta.text or resposta.caption or ""
    if not afirmacao:
        afirmacao = remover_mencoes(message.text, entidades)

    if not afirmacao:
        await message.reply_text(
            "Me marque junto com a afirmação que quer checar, assim:\n\n"
            f"@{context.bot.username} O RU vai fechar em outubro?"
        )
        return

    afirmacao = scrub(afirmacao)
    veredito = await asyncio.to_thread(verificar_afirmacao, afirmacao)
    texto = formatar_veredito(veredito)
    await message.reply_text(
        texto,
        parse_mode=ParseMode.MARKDOWN_V2,
        link_preview_options=LinkPreviewOptions(is_disabled=True),
        reply_markup=montar_teclado_fontes(veredito),
    )


async def mostrar_fontes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    token = (query.data or "").removeprefix("fontes:")
    fontes = _fontes_por_token.pop(token, None)

    if fontes is None:
        await query.answer("As fontes dessa checagem não estão mais disponíveis.", show_alert=True)
        return

    await query.answer()
    texto_atual = query.message.text_markdown_v2 or query.message.text or ""
    novo_texto = f"{texto_atual}\n\n{formatar_fontes(fontes)}"
    await query.edit_message_text(
        novo_texto,
        parse_mode=ParseMode.MARKDOWN_V2,
        link_preview_options=LinkPreviewOptions(is_disabled=True),
    )


async def checar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    afirmacao = extrair_afirmacao(update, context)
    if not afirmacao:
        await update.message.reply_text(
            "Me diga o que checar. Cole a afirmação suspeita junto do "
            "comando, assim:\n\n"
            "/checar O RU vai fechar em outubro?"
        )
        return

    afirmacao = scrub(afirmacao)
    veredito = await asyncio.to_thread(verificar_afirmacao, afirmacao)
    texto = formatar_veredito(veredito)
    await update.message.reply_text(
        texto,
        parse_mode=ParseMode.MARKDOWN_V2,
        link_preview_options=LinkPreviewOptions(is_disabled=True),
        reply_markup=montar_teclado_fontes(veredito),
    )


async def encerrar_observabilidade(app: Application) -> None:
    """Envia ao Langfuse os traces que ainda estão na fila antes do processo sair."""
    observability.encerrar()


async def tratar_erro(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Erro ao processar um update", exc_info=context.error)


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN não foi declarado ou está expirado")

    app = Application.builder().token(token).post_shutdown(encerrar_observabilidade).build()

    apenas_mensagens_novas = filters.UpdateType.MESSAGE

    app.add_handler(CommandHandler("start", start, filters=apenas_mensagens_novas))
    app.add_handler(CommandHandler("help", ajuda, filters=apenas_mensagens_novas))
    app.add_handler(CommandHandler("privacidade", privacidade, filters=apenas_mensagens_novas))
    app.add_handler(CommandHandler("checar", checar, filters=apenas_mensagens_novas))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND & apenas_mensagens_novas, mencao)
    )
    app.add_handler(CallbackQueryHandler(mostrar_fontes, pattern=r"^fontes:"))
    app.add_error_handler(tratar_erro)

    if app.job_queue is not None:
        app.job_queue.run_repeating(
            ingestao_periodica,
            interval=INTERVALO_INGESTAO_MIN * 60,
            first=60,
            name="ingestao_periodica",
        )
        logger.info("Ingestão periódica agendada a cada %d minuto(s).", INTERVALO_INGESTAO_MIN)
    else:
        logger.warning("JobQueue indisponível; ingestão periódica desativada.")

    logger.info("Bot iniciado. Ctrl+C para parar.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
