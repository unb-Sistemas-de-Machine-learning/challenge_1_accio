import asyncio
import logging
from datetime import UTC, datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from .crawler import load_known_urls, run_crawler
from .rss import fetch_unb_rss_feed

logger = logging.getLogger(__name__)


async def run_rss_ingestion(output_file: str):
    logger.info("Iniciando task de ingestão via RSS...")
    known_urls = load_known_urls(output_file)
    docs = await asyncio.to_thread(fetch_unb_rss_feed)

    with open(output_file, "a", encoding="utf-8") as f:
        for doc in docs:
            if doc.url not in known_urls:
                f.write(doc.model_dump_json() + "\n")
                known_urls.add(doc.url)
                logger.debug("RSS salvo: %s", doc.url)

    logger.info("Task de ingestão via RSS concluída.")


async def pipeline_job(output_file: str = "dados.txt"):
    logger.info("--- INICIANDO NOVO CICLO DE INGESTÃO: %s ---", output_file)

    try:
        await run_crawler(output_file=output_file)
    except Exception:
        logger.exception("Falha na coleta web; o ciclo continuará com o RSS.")

    try:
        await run_rss_ingestion(output_file=output_file)
    except Exception:
        logger.exception(
            "Falha na ingestão RSS; uma nova tentativa ocorrerá no próximo ciclo."
        )

    logger.info("--- CICLO DE INGESTÃO FINALIZADO: %s ---", output_file)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Configurando agendamento horário para atualizações...")
    scheduler = AsyncIOScheduler(
        timezone=UTC,
        job_defaults={
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": 300,
        },
    )

    scheduler.add_job(
        pipeline_job,
        "interval",
        id="hourly_ingestion",
        hours=1,
        kwargs={"output_file": "dados.txt"},
        next_run_time=datetime.now(tz=UTC),
        replace_existing=True,
    )

    scheduler.start()
    logger.info("Scheduler em execução. Pressione Ctrl+C para encerrar.")

    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler encerrado.")


if __name__ == "__main__":
    asyncio.run(main())
