import asyncio
import logging
import os

from dotenv import load_dotenv
from telegram import LinkPreviewOptions, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.helpers import escape_markdown

from fato_unb.bots.privacy import FiltroPII, scrub
from fato_unb.llm.checker import FactChecker
from fato_unb.llm.clients import LLMError, criar_cliente
from fato_unb.rag.models import VereditoJSON, VereditoType
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

EMOJI_VEREDITO = {
    VereditoType.CONFIRMADO_OFICIALMENTE: "✅",
    VereditoType.BOATO_SEM_REGISTRO: "❌",
    VereditoType.DESATUALIZADO_OU_FORA_DE_CONTEXTO: "⚠️",
    VereditoType.INCONCLUSIVO: "❓",
}

_fact_checker: FactChecker | None = None


def obter_fact_checker() -> FactChecker:
    global _fact_checker
    if _fact_checker is None:
        llm = criar_cliente()
        _fact_checker = FactChecker(Retriever.from_env(), llm)
    return _fact_checker


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
    emoji = EMOJI_VEREDITO[veredito.veredito]
    rotulo = veredito.veredito.value.replace("_", " ").title()

    linhas = [
        f"{emoji} *{escape_markdown(rotulo, version=2)}*",
        "",
        escape_markdown(veredito.justificativa, version=2),
        "",
        f"Confiança: {escape_markdown(f'{veredito.confianca:.0%}', version=2)}",
    ]

    if veredito.fontes:
        linhas.append("")
        linhas.append("*Fontes:*")
        for fonte in veredito.fontes:
            titulo = escape_markdown(fonte.title, version=2)
            url = escape_markdown(str(fonte.url), version=2, entity_type="text_link")
            linhas.append(f"• [{titulo}]({url})")

    return "\n".join(linhas)


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
    )


async def tratar_erro(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Erro ao processar um update", exc_info=context.error)


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN não foi declarado ou está expirado")

    app = Application.builder().token(token).build()

    apenas_mensagens_novas = filters.UpdateType.MESSAGE

    app.add_handler(CommandHandler("start", start, filters=apenas_mensagens_novas))
    app.add_handler(CommandHandler("help", ajuda, filters=apenas_mensagens_novas))
    app.add_handler(CommandHandler("privacidade", privacidade, filters=apenas_mensagens_novas))
    app.add_handler(CommandHandler("checar", checar, filters=apenas_mensagens_novas))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND & apenas_mensagens_novas, mencao)
    )
    app.add_error_handler(tratar_erro)

    logger.info("Bot iniciado. Ctrl+C para parar.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
