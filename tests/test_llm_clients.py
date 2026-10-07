import pytest

from fato_unb.llm.clients import DeepSeekClient, LLMError, criar_cliente


class _Resposta:
    def __init__(self, status=200, corpo=None, texto=""):
        self.status_code = status
        self._corpo = corpo or {}
        self.text = texto

    def json(self):
        return self._corpo


def test_deepseek_sem_chave(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(LLMError, match="DEEPSEEK_API_KEY"):
        DeepSeekClient()


def test_criar_cliente_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-teste")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    cliente = criar_cliente(provider="deepseek")
    assert cliente.nome == "deepseek:deepseek-flash"


def test_deepseek_gerar(monkeypatch):
    import requests

    capturado = {}

    def falso_post(url, headers, json, timeout):
        capturado.update(url=url, headers=headers, json=json)
        return _Resposta(corpo={
            "choices": [{"message": {"content": '{"veredito": "VERDADEIRO"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        })

    monkeypatch.setattr(requests, "post", falso_post)
    resp = DeepSeekClient(api_key="sk-teste").gerar("sistema JSON", "usuario")

    assert resp.texto == '{"veredito": "VERDADEIRO"}'
    assert (resp.tokens_entrada, resp.tokens_saida) == (10, 4)
    assert capturado["headers"]["Authorization"] == "Bearer sk-teste"
    assert capturado["json"]["response_format"] == {"type": "json_object"}
    assert capturado["json"]["messages"][0] == {"role": "system", "content": "sistema JSON"}
    # o deepseek-flash raciocina por padrão (caro, lento e pode estourar o max_tokens): o cliente desliga
    assert capturado["json"]["thinking"] == {"type": "disabled"}


def test_deepseek_raciocinio_pode_ser_ligado_por_variavel(monkeypatch):
    import requests

    capturado = {}

    def falso_post(url, headers, json, timeout):
        capturado.update(json=json)
        return _Resposta(corpo={"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr(requests, "post", falso_post)
    monkeypatch.setenv("DEEPSEEK_THINKING", "enabled")
    DeepSeekClient(api_key="sk-teste").gerar("s", "u")
    assert capturado["json"]["thinking"] == {"type": "enabled"}


def test_deepseek_saldo_insuficiente(monkeypatch):
    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resposta(status=402, texto="Insufficient Balance"))
    with pytest.raises(LLMError, match="saldo insuficiente") as exc:
        DeepSeekClient(api_key="sk-teste").gerar("s", "u")
    assert exc.value.transitorio is False


def test_deepseek_erro_transitorio(monkeypatch):
    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resposta(status=503, texto="busy"))
    with pytest.raises(LLMError) as exc:
        DeepSeekClient(api_key="sk-teste").gerar("s", "u")
    assert exc.value.transitorio is True
