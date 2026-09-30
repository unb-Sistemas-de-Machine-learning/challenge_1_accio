from datetime import UTC, datetime

import pytest

from fato_unb.evaluation.dataset import (
    DEFAULT_CORPUS,
    CasoTeste,
    TipoCaso,
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
