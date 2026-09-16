import asyncio
from datetime import datetime
import json
import logging
import os
from typing import List, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from fato_unb.ingestion.crawler import load_known_urls, run_crawler
from fato_unb.ingestion.models import RawDocument
from fato_unb.ingestion.rss import fetch_unb_rss_feed
from fato_unb.rag.pipeline import IndexingPipeline, IndexingReport
from fato_unb.storage import StagingRepository, init_db

logger = logging.getLogger(__name__)


async def run_rss_ingestion(
    output_file: Optional[str] = "dados.txt",
    repository: Optional[StagingRepository] = None,
) -> List[RawDocument]:
    logger.info("Iniciando task de ingestão via RSS...")
    repo = repository or StagingRepository()
    docs = fetch_unb_rss_feed()

    if docs:
        saved_count = await repo.save_documents(docs)
        logger.info(
            f"RSS: {len(docs)} documentos coletados, {saved_count} novos persistidos no StagingRepository."
        )

    if output_file:
        known_urls = load_known_urls(output_file)
        with open(output_file, "a", encoding="utf-8") as f:
            for doc in docs:
                url_str = str(doc.url)
                if url_str not in known_urls:
                    f.write(doc.model_dump_json() + "\n")
                    known_urls.add(url_str)
                    logger.debug(f"RSS salvo em arquivo: {url_str}")

    logger.info("Task de ingestão via RSS concluída.")
    return docs


async def pipeline_job(
    output_file: str = "dados.txt",
    repository: Optional[StagingRepository] = None,
    pipeline: Optional[IndexingPipeline] = None,
    max_crawler_pages: Optional[int] = 25,
) -> IndexingReport:
    logger.info(f"--- INICIANDO NOVO CICLO DE INGESTÃO: {output_file} ---")

    await init_db()
    repo = repository or StagingRepository()

    await run_crawler(output_file=output_file, repository=repo, max_pages=max_crawler_pages)
    await run_rss_ingestion(output_file=output_file, repository=repo)

    logger.info("--- INICIANDO CICLO DE INDEXAÇÃO VETORIAL ---")
    indexing_pipeline = pipeline or IndexingPipeline(repository=repo)
    report = await indexing_pipeline.run()
    logger.info(f"--- RELATÓRIO DE INDEXAÇÃO VETORIAL: {report} ---")

    logger.info(f"--- CICLO DE INGESTÃO E INDEXAÇÃO FINALIZADO: {output_file} ---")
    return report


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Configurando agendamento horário para atualizações...")
    scheduler = AsyncIOScheduler()

    scheduler.add_job(
        pipeline_job,
        "interval",
        hours=1,
        kwargs={"output_file": "dados.txt"},
        next_run_time=datetime.now(),
    )

    scheduler.start()
    logger.info("Scheduler em execução. Pressione Ctrl+C para encerrar.")

    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Encerrando o Scheduler...")


if __name__ == "__main__":
    asyncio.run(main())