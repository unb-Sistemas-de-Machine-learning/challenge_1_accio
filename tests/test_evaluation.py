from datetime import UTC, datetime

import pytest

from fato_unb.evaluation.dataset import (
    DEFAULT_CORPUS,
    CasoTeste,
    TipoCaso,
    contains_span,
    content_key,
    dedupe_corpus,
    expected_content_keys,
    load_corpus,
    load_dataset,
    normalize_url,
    url_to_content_keys,
)
from fato_unb.evaluation.retrieval import (
    EvalConfig,
    avaliar,
    reciprocal_rank,
    recall_at_k,
)
from fato_unb.ingestion.models import RawDocument, SourceType
from fato_unb.rag.embeddings import EmbeddingService
from fato_unb.rag.models import VereditoType


def _doc(url: str, content: str) -> RawDocument:
    return RawDocument(
        title="t",
        content=content,
        url=url,
        source="s",
        source_type=SourceType.HTML_PAGE,
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_recall_and_reciprocal_rank():
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, {"b"}, 1) == 0.0
    assert recall_at_k(ranked, {"b"}, 3) == 1.0
    assert reciprocal_rank(ranked, {"b"}) == 0.5
    assert reciprocal_rank(ranked, {"zzz"}) == 0.0


def test_normalize_url_ignores_fragment_and_trailing_slash():
    assert normalize_url("https://x.br/a/#main") == normalize_url("https://x.br/a")


def test_expected_keys_cover_duplicate_urls_with_same_content():
    docs = [_doc("https://x.br/a", "texto"), _doc("https://x.br/a/#main", "texto")]
    caso = CasoTeste(
        id="c1", tipo=TipoCaso.VERDADEIRA, alegacao="?", expected_urls=["https://x.br/a"], categoria="x"
    )
    keys = expected_content_keys(caso, url_to_content_keys(docs))
    assert keys == {content_key(docs[1])}
    assert len(dedupe_corpus(docs)) == 1


def test_expected_url_missing_from_corpus_raises():
    caso = CasoTeste(
        id="c1", tipo=TipoCaso.VERDADEIRA, alegacao="?", expected_urls=["https://x.br/nao"], categoria="x"
    )
    with pytest.raises(KeyError):
        expected_content_keys(caso, {})


@pytest.mark.skipif(not DEFAULT_CORPUS.exists(), reason="corpus dados.txt ausente")
def test_shipped_dataset_is_consistent_with_corpus():
    casos = load_dataset()
    url_map = url_to_content_keys(load_corpus())
    assert len(casos) >= 80
    for caso in casos:
        if caso.tipo == TipoCaso.SEM_REGISTRO:
            assert caso.expected_urls == []
        else:
            assert caso.expected_urls, caso.id
            expected_content_keys(caso, url_map)  # levanta se a URL sumiu do corpus


def test_avaliar_smoke_with_mock_embedder():
    docs = [_doc("https://x.br/a", "A UnB abre matrículas em julho."), _doc("https://x.br/b", "O RU fecha no feriado.")]
    casos = [
        CasoTeste(id="c1", tipo=TipoCaso.VERDADEIRA, alegacao="matrícula", expected_urls=["https://x.br/a"], categoria="x"),
        CasoTeste(id="c2", tipo=TipoCaso.SEM_REGISTRO, alegacao="mensalidade", categoria="x"),
    ]
    res = avaliar(casos, docs, EvalConfig(name="mock"), embedder=EmbeddingService(provider="mock"))

    assert res.n_docs == 2 and res.n_chunks == 2
    assert res.agregado()["n"] == 1  # sem_registro fica fora das métricas de recall
    assert res.casos[1].rr is None
    assert set(res.agregado()) >= {"recall@1", "recall@3", "recall@5", "mrr"}


def test_shipped_dataset_verdicts_match_case_type():
    """Cada tipo de caso só admite os vereditos coerentes com ele."""
    permitidos = {
        TipoCaso.VERDADEIRA: {VereditoType.CONFIRMADO_OFICIALMENTE},
        TipoCaso.FALSA: {VereditoType.BOATO_SEM_REGISTRO, VereditoType.DESATUALIZADO_OU_FORA_DE_CONTEXTO},
        TipoCaso.DESATUALIZADA: {VereditoType.DESATUALIZADO_OU_FORA_DE_CONTEXTO},
        TipoCaso.SEM_REGISTRO: {VereditoType.BOATO_SEM_REGISTRO},
        TipoCaso.PERGUNTA: {None},
    }
    for caso in load_dataset():
        assert caso.veredito_esperado in permitidos[caso.tipo], caso.id


def test_shipped_dataset_covers_hard_cases():
    casos = load_dataset()
    desafios = {c.desafio for c in casos}
    assert {"parafrase", "coloquial", "sigla", "confundidor", "numerico", "multi_doc", "temporal"} <= desafios
    assert sum(c.dificuldade == "dificil" for c in casos) >= 30


def test_contains_span_ignores_whitespace_and_line_breaks():
    assert contains_span("O prazo\nvai até 30 de   setembro.", ["prazo vai até 30 de setembro"])
    assert not contains_span("O prazo vai até 21 de setembro.", ["30 de setembro"])
    assert not contains_span("qualquer texto", [])


@pytest.mark.skipif(not DEFAULT_CORPUS.exists(), reason="corpus dados.txt ausente")
def test_shipped_evidence_spans_exist_in_expected_documents():
    """Um span que não está no documento esperado tornaria a métrica por chunk silenciosamente errada."""
    docs = load_corpus()
    url_map = url_to_content_keys(docs)
    texto_por_chave = {content_key(d): d.content for d in docs}
    for caso in load_dataset():
        if caso.tipo == TipoCaso.SEM_REGISTRO:
            assert caso.evidence_spans == [], caso.id
            continue
        assert caso.evidence_spans, f"{caso.id} sem evidence_spans"
        textos = [texto_por_chave[k] for k in expected_content_keys(caso, url_map)]
        for span in caso.evidence_spans:
            assert any(contains_span(t, [span]) for t in textos), f"{caso.id}: '{span}' não está no documento"


def test_chunk_and_context_metrics_distinguish_chunk_from_parent():
    """Com o chunk pequeno a evidência pode ficar fora do chunk mas dentro do parent_text."""
    frases = [f"Frase {i} sem relevancia alguma para o assunto tratado." for i in range(30)]
    frases[15] = "O prazo final de inscricao e 30 de setembro."
    doc = _doc("https://x.br/a", " ".join(frases))
    caso = CasoTeste(
        id="c1",
        tipo=TipoCaso.VERDADEIRA,
        alegacao="prazo final de inscricao",
        expected_urls=["https://x.br/a"],
        categoria="x",
        evidence_spans=["O prazo final de inscricao e 30 de setembro"],
    )
    from fato_unb.rag.chunker import SemanticChunker

    chunks = SemanticChunker(chunk_size=20, overlap_sentences=0, parent_size=200).chunk_document(doc)
    # o chunk com a evidência contém o span; um chunk vizinho não, mas o parent dele sim
    com = [c for c in chunks if contains_span(c.raw_text, caso.evidence_spans)]
    vizinhos = [c for c in chunks if not contains_span(c.raw_text, caso.evidence_spans)
                and contains_span(c.parent_text or "", caso.evidence_spans)]
    assert len(com) == 1
    assert vizinhos, "o parent_text deveria cobrir a evidência também em chunks vizinhos"


def test_avaliar_reports_chunk_metrics_with_mock_embedder():
    docs = [_doc("https://x.br/a", "A UnB abre matrículas em julho."), _doc("https://x.br/b", "O RU fecha no feriado.")]
    caso = CasoTeste(
        id="c1", tipo=TipoCaso.VERDADEIRA, alegacao="matrícula", expected_urls=["https://x.br/a"],
        categoria="x", evidence_spans=["abre matrículas em julho"],
    )
    res = avaliar([caso], docs, EvalConfig(name="mock", chunker="sentence", chunk_size=50, overlap=0),
                  embedder=EmbeddingService(provider="mock"))
    g = res.agregado()
    assert {"chunk@1", "chunk@3", "chunk@5", "ctx@3", "chunk_mrr", "ctx_words"} <= set(g)
    assert g["chunk@5"] == 1.0 and g["ctx@5"] == 1.0  # só 2 docs; a evidência está nos 5 primeiros
