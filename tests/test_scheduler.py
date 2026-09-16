import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from fato_unb.ingestion.crawler import run_crawler
from fato_unb.ingestion.models import RawDocument, SourceType
from fato_unb.ingestion.scheduler import pipeline_job, run_rss_ingestion
from fato_unb.rag.pipeline import IndexingPipeline, IndexingReport
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
async def test_run_rss_ingestion_persists_to_staging_and_file(tmp_path, test_repo):
    sample_docs = [
        RawDocument(
            title="Notícia 1",
            content="Conteúdo da primeira notícia",
            url="https://noticias.unb.br/noticia-1",
            source="UnB Notícias",
            source_type=SourceType.RSS_NEWS,
            published_at=datetime.now(UTC),
        ),
        RawDocument(
            title="Notícia 2",
            content="Conteúdo da segunda notícia",
            url="https://noticias.unb.br/noticia-2",
            source="UnB Notícias",
            source_type=SourceType.RSS_NEWS,
            published_at=datetime.now(UTC),
        ),
    ]

    output_file = tmp_path / "dados_test.txt"

    with patch("fato_unb.ingestion.scheduler.fetch_unb_rss_feed", return_value=sample_docs):
        # 1. Primeira execução: deve persistir ambos os documentos
        returned_docs = await run_rss_ingestion(
            output_file=str(output_file), repository=test_repo
        )

        assert len(returned_docs) == 2

        pending = await test_repo.get_pending_documents(limit=10)
        assert len(pending) == 2

        lines = output_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert data["url"] in [
                "https://noticias.unb.br/noticia-1",
                "https://noticias.unb.br/noticia-2",
            ]

        # 2. Segunda execução (idempotência): não deve duplicar no repositório nem no arquivo
        returned_docs_2 = await run_rss_ingestion(
            output_file=str(output_file), repository=test_repo
        )
        assert len(returned_docs_2) == 2

        stats = await test_repo.get_stats()
        assert stats[IngestionStatus.PENDING] == 2

        lines_after = output_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines_after) == 2


@pytest.mark.anyio
async def test_run_crawler_persists_to_staging_and_file(tmp_path, test_repo):
    output_file = tmp_path / "crawler_dados.txt"

    async def mock_fetch_and_parse(session, url):
        return (
            "https://noticias.unb.br/2026/03/01/edital-graduacao",
            "Texto longo sobre o edital de graduação com mais de cinquenta palavras para validação correta de conteúdo mínimo necessário pelo filtro do crawler UnB. "
            * 3,
            [],
            "Edital de Graduação",
            datetime.now(UTC),
        )

    with patch(
        "fato_unb.ingestion.crawler.START_URLS",
        ["https://noticias.unb.br/2026/03/01/edital-graduacao"],
    ), patch("fato_unb.ingestion.crawler.fetch_and_parse", side_effect=mock_fetch_and_parse), patch(
        "fato_unb.ingestion.crawler.asyncio.sleep", new_callable=AsyncMock
    ):
        collected = await run_crawler(
            output_file=str(output_file),
            repository=test_repo,
            max_pages=1,
        )

        assert len(collected) == 1
        assert collected[0].title == "Edital de Graduação"

        pending = await test_repo.get_pending_documents()
        assert len(pending) == 1
        assert pending[0].url == "https://noticias.unb.br/2026/03/01/edital-graduacao"

        lines = output_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1


@pytest.mark.anyio
async def test_pipeline_job_end_to_end_orchestration(tmp_path, test_repo):
    output_file = tmp_path / "pipeline_job_out.txt"

    mock_pipeline = AsyncMock(spec=IndexingPipeline)
    mock_pipeline.run.return_value = IndexingReport(
        total_processed=3, total_indexed=3, total_chunks=6, total_failed=0
    )

    with patch("fato_unb.ingestion.scheduler.init_db", new_callable=AsyncMock) as mock_init_db, \
         patch("fato_unb.ingestion.scheduler.run_crawler", new_callable=AsyncMock) as mock_crawler, \
         patch("fato_unb.ingestion.scheduler.run_rss_ingestion", new_callable=AsyncMock) as mock_rss:

        report = await pipeline_job(
            output_file=str(output_file),
            repository=test_repo,
            pipeline=mock_pipeline,
            max_crawler_pages=5,
        )

        mock_init_db.assert_awaited_once()
        mock_crawler.assert_awaited_once_with(
            output_file=str(output_file), repository=test_repo, max_pages=5
        )
        mock_rss.assert_awaited_once_with(
            output_file=str(output_file), repository=test_repo
        )
        mock_pipeline.run.assert_awaited_once()

        assert report.total_processed == 3
        assert report.total_indexed == 3
        assert report.total_chunks == 6
        assert report.total_failed == 0
