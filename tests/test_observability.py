import json
import uuid
from datetime import date

import pytest

from fato_unb import observability
from fato_unb.llm.checker import FactChecker
from fato_unb.llm.clients import LLMError, LLMResposta
from fato_unb.llm.guardrails import GuardrailConfig
from fato_unb.llm.precos import custo_usd
from tests.test_fact_checker import RetrieverFalso, _ev

AFIRMACAO = "O bacharelado em IA oferta 60 vagas por ano"
RESPOSTA = {
    "veredito": "CONFIRMADO_OFICIALMENTE",
    "justificativa": "O curso oferta 60 vagas por ano [1].",
    "fontes": [1],
    "citacoes": ["ofertará anualmente 60 vagas"],
    "confianca": 0.9,
}


class LLMComUso:
    nome = "deepseek:deepseek-chat"

    def __init__(self, *respostas):
        self.respostas = list(respostas)

    def gerar(self, system, user):
        r = self.respostas.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _ok():
    return LLMResposta(
        texto=json.dumps(RESPOSTA), tokens_entrada=1000, tokens_saida=100,
        modelo_real="deepseek-flash", tokens_cache=400, fingerprint="fp_abc",
    )


@pytest.fixture
def spans(monkeypatch):
    """Liga o Langfuse de verdade, mas com um exportador em memória (nada sai da máquina)."""
    from langfuse import Langfuse
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: F401
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exportador = InMemorySpanExporter()
    # public_key única por teste: o SDK reaproveita o cliente (e o exportador) de mesma chave
    monkeypatch.setattr(observability, "_tentou", True)
    monkeypatch.setattr(
        observability, "_lf",
        Langfuse(
            public_key=f"pk-lf-{uuid.uuid4().hex}", secret_key="sk-lf-teste", base_url="http://localhost:9",
            span_exporter=exportador, mask=observability._mascarar, flush_at=1,
        ),
    )
    monkeypatch.delenv("LANGFUSE_CAPTURE_CONTENT", raising=False)

    def coletar():
        observability.flush()
        return {s.name: s for s in exportador.get_finished_spans()}

    return coletar


def _checker(llm):
    return FactChecker(RetrieverFalso([_ev()]), llm, GuardrailConfig(), hoje=date(2026, 9, 30), dormir=lambda s: None)


# ------------------------------------------------------------------ custo
def test_custo_deepseek_separa_cache_de_entrada_e_saida():
    c = custo_usd("deepseek:deepseek-flash", tokens_entrada=1000, tokens_saida=100, tokens_cache=400)
    assert c["input"] == pytest.approx((600 * 0.30 + 400 * 0.006) / 1e6)
    assert c["output"] == pytest.approx(100 * 1.20 / 1e6)
    assert c["total"] == pytest.approx(c["input"] + c["output"])


def test_custo_modelo_fora_da_tabela_e_none():
    assert custo_usd("falso:teste", 100, 10) is None


# ------------------------------------------------------------------ sem Langfuse
def test_sem_chaves_e_no_op_e_o_resultado_nao_muda(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr(observability, "_lf", None)
    monkeypatch.setattr(observability, "_tentou", False)
    c = _checker(LLMComUso(_ok())).verificar(AFIRMACAO)
    assert c.veredito.veredito.value == "CONFIRMADO_OFICIALMENTE"
    assert observability.cliente() is None


def test_tracing_desligado_por_variavel(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-x")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    assert observability.ativo() is False


# ------------------------------------------------------------------ privacidade
def test_conteudo_por_padrao_so_resume(monkeypatch):
    monkeypatch.delenv("LANGFUSE_CAPTURE_CONTENT", raising=False)
    texto = observability.conteudo("meu cpf é 123.456.789-09")
    assert "123" not in texto and "cpf" not in texto and texto.startswith("<")


def test_conteudo_capturado_passa_pelo_scrub(monkeypatch):
    monkeypatch.setenv("LANGFUSE_CAPTURE_CONTENT", "true")
    assert observability.conteudo("meu cpf é 123.456.789-09") == "meu cpf é [CPF]"


# ------------------------------------------------------------------ trace de uma checagem
def test_checagem_gera_trace_com_busca_llm_custo_e_modelo_real(spans):
    c = _checker(LLMComUso(_ok())).verificar(AFIRMACAO)

    # o resultado carrega as métricas para quem não usa o Langfuse
    assert c.tokens_cache == 400 and c.modelo_real == "deepseek-flash"
    assert c.custo_usd == pytest.approx(custo_usd("deepseek-flash", 1000, 100, 400)["total"])
    assert c.latencia_busca_ms is not None and c.latencia_total_ms >= c.latencia_llm_ms

    s = spans()
    assert set(s) == {"checar", "busca", "llm"}
    llm, raiz, busca = s["llm"], s["checar"], s["busca"]
    # tudo no mesmo trace, com a busca e o LLM abaixo da raiz
    assert llm.context.trace_id == raiz.context.trace_id == busca.context.trace_id
    assert llm.parent.span_id == raiz.context.span_id
    a = llm.attributes
    assert a["langfuse.observation.type"] == "generation"
    assert a["langfuse.observation.model.name"] == "deepseek-flash"  # o que o provedor respondeu, não o apelido pedido
    usage = json.loads(a["langfuse.observation.usage_details"])
    assert usage == {"input": 1000, "output": 100, "total": 1100}
    custo = json.loads(a["langfuse.observation.cost_details"])
    assert custo["total"] == pytest.approx(c.custo_usd)
    meta = {k.rsplit(".", 1)[-1]: v for k, v in raiz.attributes.items() if k.startswith("langfuse.observation.metadata.")}
    assert meta["veredito"] == "CONFIRMADO_OFICIALMENTE"


def test_trace_nao_leva_o_texto_da_afirmacao_por_padrao(spans):
    _checker(LLMComUso(_ok())).verificar(AFIRMACAO)
    tudo = " ".join(str(v) for sp in spans().values() for v in sp.attributes.values())
    assert AFIRMACAO not in tudo
    assert "caracteres, sha256" in tudo


def test_erro_do_provedor_marca_a_generation_com_erro(spans):
    c = _checker(LLMComUso(LLMError("cota estourada", transitorio=False))).verificar(AFIRMACAO)
    assert "llm_indisponivel" in c.guardrails
    llm = spans()["llm"]
    assert llm.attributes["langfuse.observation.level"] == "ERROR"
    assert "cota estourada" in llm.attributes["langfuse.observation.status_message"]


def test_falha_do_langfuse_nao_derruba_a_checagem(monkeypatch):
    class Quebrado:
        def start_as_current_observation(self, **_):
            raise RuntimeError("langfuse fora do ar")

    monkeypatch.setattr(observability, "_tentou", True)
    monkeypatch.setattr(observability, "_lf", Quebrado())
    c = _checker(LLMComUso(_ok())).verificar(AFIRMACAO)
    assert c.veredito.veredito.value == "CONFIRMADO_OFICIALMENTE"
