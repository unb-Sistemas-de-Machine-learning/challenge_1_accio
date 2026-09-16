import pytest
from datetime import datetime, UTC
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from fato_unb.ingestion.models import RawDocument, SourceType
from fato_unb.storage.db import Base, IngestionStatus
from fato_unb.storage.repository import StagingRepository


@pytest.fixture
async def test_repo():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    repo = StagingRepository(session_factory=session_factory)
    yield repo
    await engine.dispose()


@pytest.mark.anyio
async def test_save_and_get_pending_documents(test_repo):
    doc1 = RawDocument(
        title="Notícia 1",
        content="Conteúdo 1",
        url="https://noticias.unb.br/1",
        source="UnB",
        source_type=SourceType.RSS_NEWS,
        published_at=datetime.now(UTC),
    )
    doc2 = RawDocument(
        title="Notícia 2",
        content="Conteúdo 2",
        url="https://noticias.unb.br/2",
        source="UnB",
        source_type=SourceType.RSS_NEWS,
        published_at=datetime.now(UTC),
    )

    # 1. Salvar documentos
    inserted = await test_repo.save_documents([doc1, doc2])
    assert inserted == 2

    # 2. Testar deduplicação: tentar inserir doc1 novamente não insere duplicata
    inserted_duplicate = await test_repo.save_documents([doc1])
    assert inserted_duplicate == 0

    # 3. Buscar pendentes
    pending = await test_repo.get_pending_documents(limit=10)
    assert len(pending) == 2
    assert pending[0].doc_id == doc1.doc_id


@pytest.mark.anyio
async def test_mark_as_indexed_and_failed(test_repo):
    doc = RawDocument(
        title="Notícia 3",
        content="Conteúdo 3",
        url="https://noticias.unb.br/3",
        source="UnB",
        source_type=SourceType.RSS_NEWS,
        published_at=datetime.now(UTC),
    )
    await test_repo.save_documents([doc])

    # Marcar como INDEXED
    await test_repo.mark_as_indexed(doc.doc_id)
    pending = await test_repo.get_pending_documents()
    assert len(pending) == 0

    # Marcar como FAILED
    doc4 = RawDocument(
        title="Notícia 4",
        content="Conteúdo 4",
        url="https://noticias.unb.br/4",
        source="UnB",
        source_type=SourceType.RSS_NEWS,
        published_at=datetime.now(UTC),
    )
    await test_repo.save_documents([doc4])
    await test_repo.mark_as_failed(doc4.doc_id, "Erro no FastEmbed")

    stats = await test_repo.get_stats()
    assert stats[IngestionStatus.INDEXED] == 1
    assert stats[IngestionStatus.FAILED] == 1
    assert stats[IngestionStatus.PENDING] == 0


@pytest.mark.anyio
async def test_reset_failed_to_pending(test_repo):
    doc = RawDocument(
        title="Notícia Falha",
        content="Conteúdo Falha",
        url="https://noticias.unb.br/falha",
        source="UnB",
        source_type=SourceType.RSS_NEWS,
        published_at=datetime.now(UTC),
    )
    await test_repo.save_documents([doc])
    await test_repo.mark_as_failed(doc.doc_id, "Erro temporário")

    stats_before = await test_repo.get_stats()
    assert stats_before[IngestionStatus.FAILED] == 1
    assert stats_before[IngestionStatus.PENDING] == 0

    reset_count = await test_repo.reset_failed_to_pending()
    assert reset_count == 1

    stats_after = await test_repo.get_stats()
    assert stats_after[IngestionStatus.FAILED] == 0
    assert stats_after[IngestionStatus.PENDING] == 1


@pytest.mark.anyio
async def test_save_documents_intra_batch_dedup_and_empty(test_repo):
    # Salvar lista vazia
    assert await test_repo.save_documents([]) == 0

    doc = RawDocument(
        title="Duplicata em lote",
        content="Conteúdo Duplicata",
        url="https://noticias.unb.br/dup",
        source="UnB",
        source_type=SourceType.RSS_NEWS,
        published_at=datetime.now(UTC),
    )
    # Lista com o mesmo documento repetido
    inserted = await test_repo.save_documents([doc, doc])
    assert inserted == 1

