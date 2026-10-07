from types import SimpleNamespace

import pytest

from fato_unb.bots.privacy import scrub
from fato_unb.llm.clients import LLMError
from fato_unb.bots.telegram_bot import (
    entidades_mencao_ao_bot,
    extrair_afirmacao,
    formatar_fontes,
    formatar_veredito,
    montar_teclado_fontes,
    remover_mencoes,
    verificar_afirmacao_provisoria_com_contextos,
)
from fato_unb.rag.models import FonteCitada, VereditoJSON, VereditoType


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("meu cpf é 123.456.789-00", "meu cpf é [CPF]"),
        ("meu cpf é 12345678900", "meu cpf é [CPF]"),
        ("minha matrícula é 230012345", "minha matrícula é [MATRICULA]"),
        ("me liga no (61) 99123-4567", "me liga no [TELEFONE]"),
        ("me liga no 6132251100", "me liga no [TELEFONE]"),
        ("me manda email para aluno@aluno.unb.br", "me manda email para [EMAIL]"),
        ("o RU vai fechar em outubro?", "o RU vai fechar em outubro?"),
    ],
)
def test_scrub_um_dado_por_vez(texto, esperado):
    assert scrub(texto) == esperado


def test_scrub_varios_dados_misturados():
    texto = "Sou 123.456.789-00, matrícula 230012345, me liga (61) 99123-4567 ou manda pra aluno@unb.br"
    resultado = scrub(texto)

    assert "123.456.789-00" not in resultado
    assert "230012345" not in resultado
    assert "99123-4567" not in resultado
    assert "aluno@unb.br" not in resultado
    assert "[CPF]" in resultado
    assert "[MATRICULA]" in resultado
    assert "[TELEFONE]" in resultado
    assert "[EMAIL]" in resultado


def test_scrub_telefone_de_11_digitos_sem_formatacao_e_tratado_como_cpf():
    assert scrub("me liga no 61991234567") == "me liga no [CPF]"


def test_scrub_texto_vazio_nao_quebra():
    assert scrub("") == ""
    assert scrub(None) is None


def _fake_update(texto_comando=None, reply_texto=None, reply_caption=None):
    reply = None
    if reply_texto is not None or reply_caption is not None:
        reply = SimpleNamespace(text=reply_texto, caption=reply_caption)
    message = SimpleNamespace(reply_to_message=reply)
    return SimpleNamespace(message=message)


def _fake_context(args=None):
    return SimpleNamespace(args=args or [])


def test_extrair_afirmacao_prioriza_reply():
    update = _fake_update(reply_texto="O RU vai fechar em outubro?")
    context = _fake_context(args=["outro", "texto"])

    assert extrair_afirmacao(update, context) == "O RU vai fechar em outubro?"


def test_extrair_afirmacao_usa_args_sem_reply():
    update = _fake_update()
    context = _fake_context(args=["O", "RU", "vai", "fechar?"])

    assert extrair_afirmacao(update, context) == "O RU vai fechar?"


def test_extrair_afirmacao_usa_caption_do_reply():
    update = _fake_update(reply_caption="legenda da imagem suspeita")
    context = _fake_context()

    assert extrair_afirmacao(update, context) == "legenda da imagem suspeita"


def test_extrair_afirmacao_vazio_sem_reply_e_sem_args():
    update = _fake_update()
    context = _fake_context()

    assert extrair_afirmacao(update, context) == ""


def _fake_entidade(tipo, offset, length):
    return SimpleNamespace(type=tipo, offset=offset, length=length)


def test_entidades_mencao_ao_bot_reconhece_mencao_exata():
    texto = "@FatoUnB_bot isso é verdade?"
    message = SimpleNamespace(text=texto, entities=[_fake_entidade("mention", 0, 12)])

    entidades = entidades_mencao_ao_bot(message, "FatoUnB_bot")

    assert len(entidades) == 1


def test_entidades_mencao_ao_bot_ignora_username_parecido():
    texto = "@FatoUnB_bot_fake isso é verdade?"
    message = SimpleNamespace(text=texto, entities=[_fake_entidade("mention", 0, 17)])

    entidades = entidades_mencao_ao_bot(message, "FatoUnB_bot")

    assert entidades == []


def test_remover_mencoes_sobra_o_resto_do_texto():
    texto = "@FatoUnB_bot O RU vai fechar em outubro?"
    entidades = [_fake_entidade("mention", 0, 12)]

    assert remover_mencoes(texto, entidades) == "O RU vai fechar em outubro?"


def _veredito_teste(
    *,
    veredito=VereditoType.INCONCLUSIVO,
    justificativa="Justificativa de teste.",
    fontes=None,
):
    return VereditoJSON(
        veredito=veredito,
        justificativa=justificativa,
        fontes=fontes or [],
        confianca=0.9,
        afirmacao_analisada="Afirmação de teste.",
    )


def test_verificar_afirmacao_retorna_contextos_do_fact_checker(monkeypatch):
    veredito = _veredito_teste(
        veredito=VereditoType.CONFIRMADO_OFICIALMENTE,
        fontes=[
            FonteCitada(
                title="Notícia",
                url="https://noticias.unb.br/x",
                source="UnB Notícias",
            )
        ],
    )
    evidencias = [
        SimpleNamespace(contexto="Contexto usado pelo verificador."),
        SimpleNamespace(contexto="   "),
    ]
    checker = SimpleNamespace(
        verificar=lambda texto: SimpleNamespace(
            veredito=veredito,
            evidencias=evidencias,
        )
    )
    monkeypatch.setattr(
        "fato_unb.bots.telegram_bot.obter_fact_checker",
        lambda: checker,
    )

    actual_verdict, contexts = verificar_afirmacao_provisoria_com_contextos(
        "Afirmação"
    )

    assert actual_verdict is veredito
    assert contexts == ["Contexto usado pelo verificador."]


def test_verificar_afirmacao_sem_cliente_llm_retorna_inconclusivo(monkeypatch):
    def raise_missing_key():
        raise LLMError("Chave ausente.")

    monkeypatch.setattr(
        "fato_unb.bots.telegram_bot.obter_fact_checker",
        raise_missing_key,
    )

    verdict, contexts = verificar_afirmacao_provisoria_com_contextos("Afirmação")

    assert verdict.veredito == VereditoType.INCONCLUSIVO
    assert contexts == []


def test_verificar_afirmacao_com_raise_on_error_propaga_falha_llm(monkeypatch):
    error = LLMError("Chave ausente.")

    def raise_error():
        raise error

    monkeypatch.setattr(
        "fato_unb.bots.telegram_bot.obter_fact_checker",
        raise_error,
    )

    with pytest.raises(LLMError) as exc_info:
        verificar_afirmacao_provisoria_com_contextos(
            "Afirmação",
            raise_on_error=True,
        )

    assert exc_info.value is error


def test_formatar_veredito_escapa_markdown():
    texto = formatar_veredito(_veredito_teste(justificativa="teste."))

    assert "\\." in texto
    assert "Confiança:" in texto


@pytest.mark.parametrize(
    ("verdict_type", "emoji"),
    [
        (VereditoType.CONFIRMADO_OFICIALMENTE, "✅"),
        (VereditoType.BOATO_SEM_REGISTRO, "❌"),
        (VereditoType.DESATUALIZADO_OU_FORA_DE_CONTEXTO, "⚠️"),
        (VereditoType.INCONCLUSIVO, "❓"),
    ],
)
def test_formatar_veredito_suporta_todos_os_estados(verdict_type, emoji):
    assert formatar_veredito(
        _veredito_teste(veredito=verdict_type)
    ).startswith(f"{emoji} ")


def test_formatar_veredito_nao_inclui_fontes_no_corpo():
    veredito = VereditoJSON(
        veredito=VereditoType.CONFIRMADO_OFICIALMENTE,
        justificativa="teste",
        fontes=[FonteCitada(title="Notícia", url="https://noticias.unb.br/x", source="UnB Notícias")],
        confianca=0.9,
        afirmacao_analisada="teste",
    )
    texto = formatar_veredito(veredito)

    assert "Fontes" not in texto


def test_formatar_fontes_lista_titulo_e_link():
    fontes = [FonteCitada(title="Notícia X", url="https://noticias.unb.br/x", source="UnB Notícias")]
    texto = formatar_fontes(fontes)

    assert "Fontes" in texto
    assert "Notícia X" in texto
    assert "noticias.unb.br/x" in texto


def test_montar_teclado_fontes_vazio_sem_fontes():
    veredito = VereditoJSON(
        veredito=VereditoType.INCONCLUSIVO,
        justificativa="teste",
        fontes=[],
        confianca=0.0,
        afirmacao_analisada="teste",
    )

    assert montar_teclado_fontes(veredito) is None


def test_montar_teclado_fontes_cria_botao_com_contagem():
    veredito = VereditoJSON(
        veredito=VereditoType.CONFIRMADO_OFICIALMENTE,
        justificativa="teste",
        fontes=[
            FonteCitada(title="A", url="https://noticias.unb.br/a", source="UnB Notícias"),
            FonteCitada(title="B", url="https://noticias.unb.br/b", source="UnB Notícias"),
        ],
        confianca=0.9,
        afirmacao_analisada="teste",
    )

    teclado = montar_teclado_fontes(veredito)

    assert teclado is not None
    botao = teclado.inline_keyboard[0][0]
    assert "(2)" in botao.text
    assert botao.callback_data.startswith("fontes:")
