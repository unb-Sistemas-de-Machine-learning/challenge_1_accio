import uuid
from datetime import UTC, datetime

import pytest
from qdrant_client import QdrantClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from fato_unb.ingestion.models import RawDocument, SourceType
from fato_unb.rag.chunker import SemanticChunker
from fato_unb.rag.embeddings import EmbeddingService
from fato_unb.rag.pipeline import IndexingPipeline, NAMESPACE
from fato_unb.storage.db import Base, IngestionStatus
from fato_unb.storage.repository import StagingRepository


@pytest.fixture
async def setup_pipeline():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    repo = StagingRepository(session_factory=session_factory)
    qdrant = QdrantClient(":memory:")
    embedder = EmbeddingService(provider="mock", mock_dimension=384)
    chunker = SemanticChunker(chunk_size=100, overlap_sentences=1)

    pipeline = IndexingPipeline(
        repository=repo,
        chunker=chunker,
        embedder=embedder,
        qdrant_client=qdrant,
        collection_name="test_fato_noticias",
        batch_size=32,
    )

    yield pipeline, repo, qdrant, embedder
    await engine.dispose()


@pytest.mark.anyio
async def test_indexing_pipeline_end_to_end(setup_pipeline):
    pipeline, repo, qdrant, _ = setup_pipeline

    doc = RawDocument(
        title="Comunicado da Reitoria",
        content="As aulas do primeiro semestre iniciam regularmente em todas as unidades acadêmicas.",
        url="https://unb.br/comunicado-reitoria",
        source="UnB",
        source_type=SourceType.RSS_NEWS,
        published_at=datetime.now(UTC),
        semester_ref="2026.1",
    )
    await repo.save_documents([doc])

    # Executa o pipeline
    report = await pipeline.run(max_docs=10)

    assert report.total_processed == 1
    assert report.total_indexed == 1
    assert report.total_chunks >= 1
    assert report.total_failed == 0

    # Verifica que o documento no repo está como INDEXED
    stats = await repo.get_stats()
    assert stats[IngestionStatus.INDEXED] == 1
    assert stats[IngestionStatus.PENDING] == 0

    # Verifica que o Qdrant recebeu os pontos
    points_count = qdrant.count(collection_name="test_fato_noticias").count
    assert points_count == report.total_chunks


@pytest.mark.anyio
async def test_indexing_pipeline_handles_empty_document(setup_pipeline):
    pipeline, repo, qdrant, _ = setup_pipeline

    empty_doc = RawDocument(
        title="Documento Sem Conteúdo",
        content="   ",
        url="https://unb.br/vazio",
        source="UnB",
        source_type=SourceType.HTML_PAGE,
        published_at=datetime.now(UTC),
    )
    await repo.save_documents([empty_doc])

    report = await pipeline.run(max_docs=10)
    assert report.total_processed == 1
    assert report.total_failed == 1

    stats = await repo.get_stats()
    assert stats[IngestionStatus.FAILED] == 1
    assert stats[IngestionStatus.INDEXED] == 0


@pytest.mark.anyio
async def test_indexing_pipeline_no_pending_documents(setup_pipeline):
    pipeline, repo, qdrant, _ = setup_pipeline

    report = await pipeline.run(max_docs=10)
    assert report.total_processed == 0
    assert report.total_indexed == 0
    assert report.total_chunks == 0
    assert report.total_failed == 0


@pytest.mark.anyio
async def test_indexing_pipeline_deterministic_uuids(setup_pipeline):
    pipeline, repo, qdrant, _ = setup_pipeline

    doc = RawDocument(
        title="Guia do Calouro",
        content="Seja bem-vindo à Universidade de Brasília. Conheça os serviços do campus Darcy Ribeiro.",
        url="https://unb.br/guia-calouro",
        source="UnB",
        source_type=SourceType.HTML_PAGE,
        published_at=datetime.now(UTC),
    )
    await repo.save_documents([doc])

    report = await pipeline.run(max_docs=10)
    assert report.total_indexed == 1

    points, _ = qdrant.scroll(collection_name="test_fato_noticias", limit=10, with_vectors=True)
    assert len(points) == report.total_chunks
    for pt in points:
        # Verifica se o ID é um UUID válido e determinístico
        parsed_uuid = uuid.UUID(pt.id)
        assert parsed_uuid.version == 5
        assert pt.payload["title"] == "Guia do Calouro"
        assert pt.vector is not None and "dense" in pt.vector


@pytest.mark.anyio
async def test_indexing_pipeline_exception_handling(setup_pipeline):
    pipeline, repo, qdrant, embedder = setup_pipeline

    doc = RawDocument(
        title="Documento com Falha de Embedder",
        content="Conteúdo válido para gerar chunks.",
        url="https://unb.br/falha",
        source="UnB",
        source_type=SourceType.HTML_PAGE,
        published_at=datetime.now(UTC),
    )
    await repo.save_documents([doc])

    # Força exceção no embedder
    def fail_embed(texts):
        raise RuntimeError("Falha proposital de rede no embedder")

    embedder.embed_texts = fail_embed

    report = await pipeline.run(max_docs=10)
    assert report.total_processed == 1
    assert report.total_failed == 1
    assert report.total_indexed == 0

    stats = await repo.get_stats()
    assert stats[IngestionStatus.FAILED] == 1
