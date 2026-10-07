from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from fato_unb.ingestion.models import RawDocument, SourceType
from fato_unb.ingestion.rss import enrich_documents_with_full_text, enrich_new_documents
from fato_unb.ingestion.scheduler import run_rss_ingestion
from fato_unb.storage.db import Base
from fato_unb.storage.repository import StagingRepository

RESUMO = "Resumo curto da noticia do feed."
TEXTO_COMPLETO = "Texto completo da noticia. " * 40


def _doc(n: int, content: str = RESUMO) -> RawDocument:
    return RawDocument(
        title=f"Noticia {n}",
        content=content,
        url=f"https://noticias.unb.br/ensino/{n}-noticia",
        source="UnB Notícias",
        source_type=SourceType.RSS_NEWS,
        published_at=datetime(2026, 9, 1, tzinfo=UTC),
    )


@pytest.fixture
async def repo():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield StagingRepository(session_factory=session_factory)
    await engine.dispose()


def test_summary_is_replaced_by_full_text():
    docs = [_doc(1)]
    with patch("fato_unb.ingestion.rss.extract_html_data", return_value={"content": TEXTO_COMPLETO}):
        assert enrich_documents_with_full_text(docs) == 1
    assert docs[0].content == TEXTO_COMPLETO.strip()


def test_summary_is_kept_when_download_fails():
    docs = [_doc(1)]
    with patch("fato_unb.ingestion.rss.extract_html_data", side_effect=ValueError("Falha no download")):
        assert enrich_documents_with_full_text(docs) == 0
    assert docs[0].content == RESUMO


def test_summary_is_kept_when_extracted_text_is_not_longer():
    docs = [_doc(1)]
    with patch("fato_unb.ingestion.rss.extract_html_data", return_value={"content": "curto"}):
        assert enrich_documents_with_full_text(docs) == 0
    assert docs[0].content == RESUMO


def test_one_failure_does_not_stop_the_others():
    docs = [_doc(1), _doc(2)]
    resultados = [ValueError("boom"), {"content": TEXTO_COMPLETO}]
    with patch("fato_unb.ingestion.rss.extract_html_data", side_effect=resultados):
        assert enrich_documents_with_full_text(docs) == 1
    assert docs[0].content == RESUMO and docs[1].content == TEXTO_COMPLETO.strip()


@pytest.mark.anyio
async def test_only_documents_new_to_staging_are_downloaded(repo):
    ja_existe, novo = _doc(1), _doc(2)
    await repo.save_documents([ja_existe])

    with patch("fato_unb.ingestion.rss.extract_html_data", return_value={"content": TEXTO_COMPLETO}) as extract:
        assert await enrich_new_documents([ja_existe, novo], repo) == 1

    extract.assert_called_once_with(str(novo.url))
    assert ja_existe.content == RESUMO  # não foi baixado de novo
    assert novo.content == TEXTO_COMPLETO.strip()


@pytest.mark.anyio
async def test_run_rss_ingestion_persists_full_text_and_can_skip_it(repo, tmp_path):
    with patch("fato_unb.ingestion.scheduler.fetch_unb_rss_feed", return_value=[_doc(1)]), patch(
        "fato_unb.ingestion.rss.extract_html_data", return_value={"content": TEXTO_COMPLETO}
    ):
        await run_rss_ingestion(output_file=None, repository=repo)
    (salvo,) = await repo.get_pending_documents(limit=5)
    assert salvo.content == TEXTO_COMPLETO.strip()

    with patch("fato_unb.ingestion.scheduler.fetch_unb_rss_feed", return_value=[_doc(2)]), patch(
        "fato_unb.ingestion.rss.extract_html_data"
    ) as extract:
        docs = await run_rss_ingestion(output_file=None, repository=repo, fetch_full_text=False)
    extract.assert_not_called()
    assert docs[0].content == RESUMO


@pytest.mark.anyio
async def test_update_content_and_requeue_marks_document_pending_again(repo):
    rss, html = _doc(1), _doc(2)
    html.source_type = SourceType.HTML_PAGE
    await repo.save_documents([rss, html])
    await repo.mark_as_indexed(rss.doc_id)
    assert await repo.get_pending_documents(limit=10) == [
        d for d in await repo.get_pending_documents(limit=10) if d.doc_id == html.doc_id
    ]

    (so_rss,) = await repo.get_documents_by_source_type(SourceType.RSS_NEWS)
    assert so_rss.doc_id == rss.doc_id

    await repo.update_content_and_requeue(rss.doc_id, TEXTO_COMPLETO)

    pendentes = {d.doc_id: d for d in await repo.get_pending_documents(limit=10)}
    assert pendentes[rss.doc_id].content == TEXTO_COMPLETO
