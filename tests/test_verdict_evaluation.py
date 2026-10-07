import json
from datetime import UTC, date, datetime

import pytest

from fato_unb.evaluation.dataset import CasoTeste, TipoCaso
from fato_unb.evaluation.verdicts import (
    ResultadoVeredito,
    calcular_metricas,
    executar,
    formatar_relatorio,
    urls_esperadas,
)
from fato_unb.ingestion.models import RawDocument, SourceType
from fato_unb.llm.clients import LLMResposta
from fato_unb.rag.models import VereditoType
from fato_unb.rag.retriever import Evidencia

C, B, D, I = "CONFIRMADO_OFICIALMENTE", "BOATO_SEM_REGISTRO", "DESATUALIZADO_OU_FORA_DE_CONTEXTO", "INCONCLUSIVO"


def _r(id, esperado, obtido, tipo="verdadeira", desafio="direto", chegou=True, guardrails=(), ti=1000, to=100):
    return ResultadoVeredito(
        id=id, tipo=tipo, desafio=desafio, alegacao=f"alegação {id}", esperado=esperado, obtido=obtido,
        guardrails=list(guardrails), evidencia_certa_no_contexto=chegou, tokens_entrada=ti, tokens_saida=to,
        latencia_ms=1000.0, modelo="gemini:gemini-3.1-flash-lite",
    )


def test_metrics_accuracy_abstention_and_dangerous_confirmations():
    rs = [
        _r("a", C, C),                       # acerto
        _r("b", B, B, tipo="falsa"),         # acerto
        _r("c", B, C, tipo="falsa"),         # confirmação INDEVIDA (erro grave)
        _r("d", C, I),                       # abstenção; confirmação perdida
        _r("e", D, B, tipo="desatualizada"), # erro
    ]
    m = calcular_metricas(rs)
    assert m["n_avaliados"] == 5 and m["acerto"] == 2 / 5
    assert m["inconclusivos"] == 1 and m["taxa_inconclusivo"] == pytest.approx(1 / 5)
    assert m["acerto_entre_respondidos"] == 2 / 4  # o INCONCLUSIVO sai do denominador
    assert m["confirmacoes_indevidas"] == ["c"] and m["confirmacoes_perdidas"] == ["d"]
    assert m["matriz"][B][C] == 1 and m["matriz"][D][B] == 1
    assert m["por_tipo"]["falsa"] == {"n": 2, "acerto": 0.5}
    assert m["atinge_meta"] is False  # 40% não passa de 50%


def test_metrics_meta_is_strictly_greater_than_half():
    assert calcular_metricas([_r("a", C, C), _r("b", C, B)])["atinge_meta"] is False  # exatamente 50%
    assert calcular_metricas([_r("a", C, C), _r("b", C, C), _r("c", C, B)])["atinge_meta"] is True


def test_metrics_split_accuracy_by_whether_the_right_evidence_reached_the_model():
    rs = [_r("a", C, C, chegou=True), _r("b", C, C, chegou=True), _r("c", C, B, chegou=False), _r("d", B, B, chegou=None)]
    m = calcular_metricas(rs)
    assert m["acerto_com_evidencia_certa"] == {"n": 2, "acerto": 1.0}
    assert m["acerto_sem_evidencia_certa"] == {"n": 1, "acerto": 0.0}


def test_metrics_cost_guardrails_and_unlabeled_cases_are_handled():
    rs = [_r("a", C, C, guardrails=["confianca_limitada"], ti=1_000_000, to=100_000),
          _r("b", B, B, guardrails=["confianca_limitada", "llm_repetido"], ti=0, to=0)]
    rs.append(_r("p", None, C, tipo="pergunta", ti=1000, to=100))  # sem rótulo: fora do acerto, mas gasta tokens
    m = calcular_metricas(rs)
    assert m["n_avaliados"] == 2
    assert m["guardrails"] == {"confianca_limitada": 2, "llm_repetido": 1}
    # o custo soma TODAS as chamadas, inclusive a do caso sem rótulo: 1,001M entrada a $0,25 + 100,1k saída a $1,50/M
    assert m["custo_usd"] == pytest.approx(1_001_000 * 0.25e-6 + 100_100 * 1.50e-6)


def test_report_states_the_goal_and_lists_errors_on_request():
    rs = [_r("a", C, C), _r("b", B, C, tipo="falsa", chegou=False)]
    texto = formatar_relatorio(calcular_metricas(rs), rs, falhas=True)
    assert "ACERTO GERAL" in texto and "50.0%" in texto and "NÃO atingida" in texto
    assert "confirmações INDEVIDAS" in texto and "SEM evidência" in texto and "alegação b" in texto


# ------------------------------------------------------------------ execução
def _doc(url, content="texto"):
    return RawDocument(title="t", content=content, url=url, source="noticias.unb.br",
                       source_type=SourceType.HTML_PAGE, published_at=datetime(2026, 1, 1, tzinfo=UTC))


def _caso(id, tipo, veredito, urls=(), alegacao=None):
    return CasoTeste(id=id, tipo=TipoCaso(tipo), alegacao=alegacao or f"alegação número {id} sobre a UnB",
                     expected_urls=list(urls), veredito_esperado=VereditoType(veredito) if veredito else None,
                     categoria="x")


def _ev(url, contexto="O curso oferta 60 vagas."):
    return Evidencia(doc_id="d", title="Notícia", url=url, source="noticias.unb.br", published_at="2026-03-17T10:00:00Z",
                     semester_ref=None, trecho=contexto, contexto=contexto, score=0.9, reranked=False)


class _Retriever:
    def __init__(self, por_alegacao):
        self.por_alegacao = por_alegacao

    def buscar(self, alegacao, limit=5, **_):
        return self.por_alegacao[alegacao][:limit]


class _LLM:
    nome = "falso:teste"

    def __init__(self):
        self.chamadas = 0

    def gerar(self, system, user):
        self.chamadas += 1
        r = {"veredito": B, "justificativa": "Sem registro.", "fontes": [], "citacoes": [], "confianca": 0.5}
        return LLMResposta(texto=json.dumps(r), tokens_entrada=10, tokens_saida=5)


def test_urls_esperadas_include_copies_with_the_same_content():
    docs = [_doc("https://x.br/a", "mesmo"), _doc("https://x.br/a/#main", "mesmo"), _doc("https://x.br/b", "outro")]
    caso = _caso("c1", "verdadeira", C, urls=["https://x.br/a"])
    assert urls_esperadas(caso, docs) == {"https://x.br/a"}  # '#main' normaliza para a mesma URL
    docs = [_doc("https://x.br/a", "mesmo"), _doc("https://x.br/copia", "mesmo")]
    assert urls_esperadas(caso, docs) == {"https://x.br/a", "https://x.br/copia"}


def test_execute_skips_unlabeled_questions_by_default_and_flags_missing_evidence(tmp_path):
    docs = [_doc("https://x.br/a"), _doc("https://x.br/b", "outro")]
    c1 = _caso("c1", "falsa", B, urls=["https://x.br/a"])
    c2 = _caso("c2", "falsa", B, urls=["https://x.br/a"])
    cp = _caso("c3", "pergunta", None, urls=["https://x.br/a"])
    retriever = _Retriever({c1.alegacao: [_ev("https://x.br/a")], c2.alegacao: [_ev("https://x.br/b")], cp.alegacao: []})
    rs = executar([c1, c2, cp], docs, retriever, _LLM(), hoje=date(2026, 9, 30), paralelismo=2,
                  saida=tmp_path / "v.jsonl", progresso=lambda *_: None)
    assert [r.id for r in rs] == ["c1", "c2"]  # a pergunta sem rótulo ficou de fora
    assert rs[0].evidencia_certa_no_contexto is True and rs[1].evidencia_certa_no_contexto is False
    assert rs[0].obtido == B and rs[0].modelo == "falso:teste"


def test_execute_resumes_from_the_output_file_without_calling_the_llm_again(tmp_path):
    docs = [_doc("https://x.br/a")]
    casos = [_caso("c1", "falsa", B, urls=["https://x.br/a"]), _caso("c2", "falsa", B, urls=["https://x.br/a"])]
    retriever = _Retriever({c.alegacao: [_ev("https://x.br/a")] for c in casos})
    saida = tmp_path / "v.jsonl"

    llm1 = _LLM()
    executar(casos[:1], docs, retriever, llm1, hoje=date(2026, 9, 30), saida=saida, progresso=lambda *_: None)
    assert llm1.chamadas == 1

    llm2 = _LLM()
    rs = executar(casos, docs, retriever, llm2, hoje=date(2026, 9, 30), saida=saida, progresso=lambda *_: None)
    assert llm2.chamadas == 1  # só o c2; o c1 veio do arquivo
    assert [r.id for r in rs] == ["c1", "c2"]
    assert len(saida.read_text().splitlines()) == 2


def test_metrics_separate_infrastructure_failures_from_verdict_errors():
    rs = [
        _r("a", C, C),
        _r("b", C, B),                                          # erro de veredito de verdade
        _r("c", C, I, guardrails=["llm_indisponivel"]),         # o provedor caiu: não diz nada sobre o modelo
        _r("d", B, I, guardrails=["llm_indisponivel"]),
    ]
    m = calcular_metricas(rs)
    assert m["acerto"] == 1 / 4  # conservador: conta tudo
    assert m["falhas_infra"] == 2 and m["n_sem_falhas_infra"] == 2
    assert m["acerto_sem_falhas_infra"] == 1 / 2
    texto = formatar_relatorio(m, rs)
    assert "ATENÇÃO" in texto and "INDISPONIBILIDADE DO PROVEDOR" in texto and "mede a infraestrutura" in texto


def test_report_has_no_infrastructure_warning_when_everything_answered():
    rs = [_r("a", C, C), _r("b", B, B)]
    assert "ATENÇÃO" not in formatar_relatorio(calcular_metricas(rs), rs)


def test_resume_redoes_cases_that_failed_by_infrastructure_but_keeps_real_results(tmp_path):
    docs = [_doc("https://x.br/a")]
    c1, c2 = (_caso(f"c{i}", "falsa", B, urls=["https://x.br/a"]) for i in (1, 2))
    retriever = _Retriever({c.alegacao: [_ev("https://x.br/a")] for c in (c1, c2)})
    saida = tmp_path / "v.jsonl"
    ok = _r("c1", B, B, tipo="falsa")
    falho = _r("c2", B, I, tipo="falsa", guardrails=["llm_indisponivel"])
    saida.write_text("\n".join(json.dumps(r.__dict__, ensure_ascii=False) for r in (ok, falho)) + "\n", encoding="utf-8")

    llm = _LLM()
    rs = executar([c1, c2], docs, retriever, llm, hoje=date(2026, 9, 30), saida=saida, progresso=lambda *_: None)
    assert llm.chamadas == 1  # só o c2 foi refeito
    by_id = {r.id: r for r in rs}
    assert by_id["c1"].obtido == B and by_id["c2"].obtido == B and "llm_indisponivel" not in by_id["c2"].guardrails


def test_sentence_level_attribution_separates_retrieval_misses_from_model_errors(tmp_path):
    """O documento certo pode vir e a frase certa não: nesse caso o erro é da busca, não do modelo."""
    docs = [_doc("https://x.br/a")]
    caso = _caso("c1", "verdadeira", C, urls=["https://x.br/a"])
    caso.evidence_spans = ["ofertará anualmente 60 vagas"]

    com = _ev("https://x.br/a", "Texto. O curso ofertará anualmente 60 vagas para estudantes.")
    sem = _ev("https://x.br/a", "Texto do mesmo documento, mas de outro trecho, sem o dado procurado.")

    for evidencia, esperado_flag in ((com, True), (sem, False)):
        retriever = _Retriever({caso.alegacao: [evidencia]})
        (r,) = executar([caso], docs, retriever, _LLM(), hoje=date(2026, 9, 30), progresso=lambda *_: None)
        assert r.evidencia_certa_no_contexto is True  # o DOCUMENTO veio nos dois casos
        assert r.evidencia_no_texto_enviado is esperado_flag

    m = calcular_metricas([_r("a", C, C), _r("b", C, B)])
    assert m["acerto_com_frase_enviada"]["n"] == 0  # sem a informação nos resultados, não inventa


def test_metrics_report_accuracy_split_by_sentence_reaching_the_model():
    com = _r("a", C, C)
    com.evidencia_no_texto_enviado = True
    sem = _r("b", C, B)
    sem.evidencia_no_texto_enviado = False
    m = calcular_metricas([com, sem])
    assert m["acerto_com_frase_enviada"] == {"n": 1, "acerto": 1.0}
    assert m["acerto_sem_frase_enviada"] == {"n": 1, "acerto": 0.0, "ids": ["b"]}
    assert "FRASE com a evidência" in formatar_relatorio(m, [com, sem])


def test_resume_fills_the_sentence_metric_for_old_results_without_calling_the_llm(tmp_path):
    docs = [_doc("https://x.br/a")]
    caso = _caso("c1", "falsa", B, urls=["https://x.br/a"])
    caso.evidence_spans = ["dado procurado aqui"]
    ev = _ev("https://x.br/a", "texto com o dado procurado aqui dentro")
    retriever = _Retriever({caso.alegacao: [ev]})
    saida = tmp_path / "v.jsonl"
    antigo = _r("c1", B, B, tipo="falsa")
    antigo.evidencia_no_texto_enviado = None  # resultado de uma execução anterior a esta métrica
    saida.write_text(json.dumps(antigo.__dict__, ensure_ascii=False) + "\n", encoding="utf-8")

    llm = _LLM()
    (r,) = executar([caso], docs, retriever, llm, hoje=date(2026, 9, 30), saida=saida, progresso=lambda *_: None)
    assert llm.chamadas == 0 and r.evidencia_no_texto_enviado is True
