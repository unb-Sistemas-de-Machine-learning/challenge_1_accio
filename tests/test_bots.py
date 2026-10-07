from types import SimpleNamespace

import pytest

from fato_unb.bots.privacy import scrub
from fato_unb.bots.telegram_bot import (
    entidades_mencao_ao_bot,
    extrair_afirmacao,
    formatar_veredito,
    montar_veredito_provisorio,
    remover_mencoes,
    verificar_afirmacao_provisoria_com_contextos,
)


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


def _fake_ponto(
    score,
    title="Notícia Teste",
    url="https://noticias.unb.br/teste",
    source="UnB Notícias",
    content="Conteúdo da fonte.",
):
    return SimpleNamespace(
        score=score,
        payload={
            "title": title,
            "url": url,
            "source": source,
            "content": content,
        },
    )


def test_verificar_afirmacao_retorna_os_contextos_realmente_usados(monkeypatch):
    pontos = [
        _fake_ponto(score=0.8),
        _fake_ponto(score=0.2, url="https://noticias.unb.br/irrelevante"),
    ]
    monkeypatch.setattr("fato_unb.bots.telegram_bot.obter_embedder", lambda: object())
    monkeypatch.setattr(
        "fato_unb.bots.telegram_bot.buscar",
        lambda *args, **kwargs: SimpleNamespace(points=pontos),
    )

    veredito, contextos = verificar_afirmacao_provisoria_com_contextos("Afirmação")

    assert veredito.veredito.value == "INCONCLUSIVO"
    assert len(veredito.fontes) == 1
    assert contextos == ["Conteúdo da fonte."]


def test_montar_veredito_sem_pontos_relevantes():
    veredito = montar_veredito_provisorio("O RU vai fechar em outubro?", [])

    assert veredito.afirmacao_analisada == "O RU vai fechar em outubro?"
    assert veredito.fontes == []
    assert veredito.confianca == 0.0


def test_montar_veredito_ignora_pontos_abaixo_do_limiar():
    pontos = [_fake_ponto(score=0.1)]
    veredito = montar_veredito_provisorio("teste", pontos)

    assert veredito.fontes == []
    assert veredito.confianca == 0.1


def test_montar_veredito_inclui_fontes_relevantes():
    pontos = [_fake_ponto(score=0.75, title="RU tem funcionamento normal")]
    veredito = montar_veredito_provisorio("O RU vai fechar?", pontos)

    assert len(veredito.fontes) == 1
    assert veredito.fontes[0].title == "RU tem funcionamento normal"
    assert veredito.confianca == 0.75


def test_montar_veredito_remove_fontes_duplicadas_pela_url():
    pontos = [
        _fake_ponto(score=0.8, title="Notícia X", url="https://noticias.unb.br/x"),
        _fake_ponto(score=0.7, title="Notícia X", url="https://noticias.unb.br/x"),
        _fake_ponto(score=0.6, title="Notícia Y", url="https://noticias.unb.br/y"),
    ]
    veredito = montar_veredito_provisorio("teste", pontos)

    assert len(veredito.fontes) == 2
    assert {str(f.url) for f in veredito.fontes} == {
        "https://noticias.unb.br/x",
        "https://noticias.unb.br/y",
    }


def test_formatar_veredito_escapa_markdown():
    veredito = montar_veredito_provisorio("teste.", [])
    texto = formatar_veredito(veredito)

    assert "\\." in texto
    assert "Confiança:" in texto


@pytest.mark.parametrize(
    ("veredito", "emoji"),
    [
        ("CONFIRMADO_OFICIALMENTE", "✅"),
        ("BOATO_SEM_REGISTRO", "❌"),
        ("DESATUALIZADO_OU_FORA_DE_CONTEXTO", "⚠️"),
        ("INCONCLUSIVO", "❓"),
    ],
)
def test_formatar_veredito_suporta_todos_os_estados(veredito, emoji):
    from fato_unb.rag.models import VereditoJSON, VereditoType

    resposta = VereditoJSON(
        veredito=VereditoType(veredito),
        justificativa="Justificativa.",
        confianca=0.5,
        afirmacao_analisada="Afirmação.",
    )

    assert formatar_veredito(resposta).startswith(f"{emoji} ")
