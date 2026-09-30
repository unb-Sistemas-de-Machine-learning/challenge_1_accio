from datetime import UTC, datetime

import pytest
from qdrant_client import QdrantClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from fato_unb.ingestion.models import RawDocument, SourceType
from fato_unb.rag.chunker import SemanticChunker
from fato_unb.rag.embeddings import EmbeddingService
from fato_unb.rag.reranker import Reranker, rerank_points
from fato_unb.rag.retriever import Retriever
from fato_unb.storage.db import Base, IngestionStatus
from fato_unb.storage.repository import StagingRepository
from fato_unb.vectorstore.collections import ensure_collection
from fato_unb.vectorstore.operations import buscar, upsert_documents

COLECAO = "teste_integracao"


def _doc(url: str, title: str, content: str) -> RawDocument:
    return RawDocument(
        title=title,
        content=content,
        url=url,
        source="noticias.unb.br",
        source_type=SourceType.HTML_PAGE,
        published_at=datetime(2026, 8, 1, tzinfo=UTC),
    )


@pytest.fixture
def indice():
    """Índice em memória com o embedder mock: todos os vetores densos são iguais, então só o
    reranker (ou o BM25 mock) diferencia os candidatos. Assim os testes isolam o reranker."""
    embedder = EmbeddingService(provider="mock")
    client = QdrantClient(":memory:")
    ensure_collection(client, COLECAO, embedder.vector_dimension)
    docs = [
        _doc("https://x.br/ru", "RU", "O restaurante universitario fecha no feriado de maio."),
        _doc("https://x.br/pas", "PAS", "As inscricoes do PAS vao ate 30 de setembro."),
        _doc("https://x.br/cepe", "Cepe", "O Cepe aprovou o calendario de 2027 em junho."),
    ]
    chunker = SemanticChunker()
    for d in docs:
        upsert_documents(embedder, chunker.chunk_document(d), client, COLECAO)
    return embedder, client


def test_mock_reranker_scores_by_word_overlap():
    r = Reranker(provider="mock")
    assert r.score("prazo do PAS", ["inscricoes do PAS", "cardapio do RU"]) == [2.0, 1.0]
    assert r.score("x", []) == []
    with pytest.raises(ValueError, match="não suportado"):
        Reranker(provider="invalido")


def test_rerank_points_reorders_replaces_score_and_keeps_original(indice):
    embedder, client = indice
    pontos = buscar("inscricoes PAS setembro", embedder, client, COLECAO, limit=3).points
    assert len(pontos) == 3

    reranked = rerank_points(Reranker(provider="mock"), "inscricoes PAS setembro", pontos, top_k=2)

    assert len(reranked) == 2
    assert reranked[0].payload["url"] == "https://x.br/pas"  # o de maior sobreposição
    assert reranked[0].score >= reranked[1].score
    assert "_score_busca" in reranked[0].payload  # score original preservado


def test_buscar_with_reranker_returns_at_most_limit(indice):
    embedder, client = indice
    resp = buscar(
        "calendario Cepe", embedder, client, COLECAO, limit=1, reranker=Reranker(provider="mock"), candidatos=10
    )
    assert len(resp.points) == 1
    assert resp.points[0].payload["url"] == "https://x.br/cepe"


def test_buscar_without_reranker_is_unchanged(indice):
    embedder, client = indice
    resp = buscar("qualquer coisa", embedder, client, COLECAO, limit=2)
    assert len(resp.points) == 2
    assert "_score_busca" not in resp.points[0].payload


def test_retriever_returns_distinct_pages_with_context(indice):
    embedder, client = indice
    retriever = Retriever(embedder, client, COLECAO, reranker=Reranker(provider="mock"))
    evidencias = retriever.buscar("inscricoes PAS setembro", limit=3)

    urls = [e.url for e in evidencias]
    assert urls[0] == "https://x.br/pas"
    assert len(urls) == len(set(urls))
    assert all(e.contexto for e in evidencias)
    assert evidencias[0].reranked is True


def test_retriever_discards_copies_of_the_same_page():
    embedder = EmbeddingService(provider="mock")
    client = QdrantClient(":memory:")
    ensure_collection(client, COLECAO, embedder.vector_dimension)
    texto = "O Decanato lancou o novo site institucional com navegacao acessivel."
    chunker = SemanticChunker()
    # a mesma página duas vezes, como o crawler produziu (#main), mais uma página diferente
    for url in ("https://dpg.unb.br/site/", "https://dpg.unb.br/site/#main"):
        upsert_documents(embedder, chunker.chunk_document(_doc(url, "Site DPG", texto)), client, COLECAO)
    upsert_documents(embedder, chunker.chunk_document(_doc("https://x.br/outra", "Outra", "Texto sobre outro assunto.")), client, COLECAO)

    evidencias = Retriever(embedder, client, COLECAO).buscar("novo site", limit=5)
    assert len([e for e in evidencias if "dpg.unb.br/site" in e.url]) == 1


@pytest.fixture
async def repo():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield StagingRepository(session_factory=session_factory)
    await engine.dispose()


@pytest.mark.anyio
async def test_reset_all_to_pending_requeues_indexed_and_failed(repo):
    docs = [_doc(f"https://x.br/{i}", f"t{i}", f"conteudo {i}") for i in range(3)]
    await repo.save_documents(docs)
    await repo.mark_as_indexed(docs[0].doc_id)
    await repo.mark_as_failed(docs[1].doc_id, "erro")

    assert await repo.reset_all_to_pending() == 3

    stats = await repo.get_stats()
    assert stats[IngestionStatus.PENDING] == 3
    assert stats[IngestionStatus.INDEXED] == 0 and stats[IngestionStatus.FAILED] == 0


def test_retriever_returns_limit_distinct_pages_even_if_one_document_dominates(monkeypatch):
    """Regressão: um documento com muitos chunks ocupava todos os candidatos e sobrava 1 página.

    A busca é simulada para ser determinística: os 20 primeiros resultados são do mesmo
    documento e só o 21º é de outra página."""
    from types import SimpleNamespace

    def ponto(url: str, i: int):
        payload = {"doc_id": url, "url": url, "title": url, "source": "s", "raw_text": f"{url} chunk {i}"}
        return SimpleNamespace(payload=payload, score=1.0 - i / 100)

    ranking = [ponto("https://x.br/longo", i) for i in range(20)] + [ponto("https://x.br/curto", 20)]

    def busca_simulada(*, limit, **_):
        return SimpleNamespace(points=ranking[:limit])  # como o Qdrant: só devolve `limit` pontos

    monkeypatch.setattr("fato_unb.rag.retriever.buscar", busca_simulada)
    retriever = Retriever(EmbeddingService(provider="mock"), QdrantClient(":memory:"), COLECAO)

    evidencias = retriever.buscar("PAS", limit=2)
    assert [e.url for e in evidencias] == ["https://x.br/longo", "https://x.br/curto"]
