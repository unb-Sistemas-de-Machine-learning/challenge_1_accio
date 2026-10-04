import asyncio
import threading

from fato_unb.ingestion import scheduler


def test_pipeline_continues_to_rss_when_crawler_fails(monkeypatch):
    rss_called = False

    async def failing_crawler(output_file: str) -> None:
        raise RuntimeError("portal indisponível")

    async def successful_rss(output_file: str) -> None:
        nonlocal rss_called
        rss_called = True

    monkeypatch.setattr(scheduler, "run_crawler", failing_crawler)
    monkeypatch.setattr(scheduler, "run_rss_ingestion", successful_rss)

    asyncio.run(scheduler.pipeline_job(output_file="unused.jsonl"))

    assert rss_called


def test_pipeline_swallows_rss_failure_after_crawler(monkeypatch):
    crawler_called = False

    async def successful_crawler(output_file: str) -> None:
        nonlocal crawler_called
        crawler_called = True

    async def failing_rss(output_file: str) -> None:
        raise RuntimeError("feed indisponível")

    monkeypatch.setattr(scheduler, "run_crawler", successful_crawler)
    monkeypatch.setattr(scheduler, "run_rss_ingestion", failing_rss)

    asyncio.run(scheduler.pipeline_job(output_file="unused.jsonl"))

    assert crawler_called


def test_rss_fetch_runs_off_event_loop_thread(monkeypatch, tmp_path):
    main_thread = threading.get_ident()
    fetch_thread: list[int] = []

    def fake_fetch() -> list:
        fetch_thread.append(threading.get_ident())
        return []

    monkeypatch.setattr(scheduler, "fetch_unb_rss_feed", fake_fetch)
    output_file = tmp_path / "documents.jsonl"

    asyncio.run(scheduler.run_rss_ingestion(str(output_file)))

    assert fetch_thread
    assert fetch_thread[0] != main_thread
    assert output_file.exists()
