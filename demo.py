import asyncio
import logging
import sys
from pathlib import Path

# Garante importação do pacote fato_unb se executado a partir da raiz
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fato_unb.ingestion.scheduler import pipeline_job
from fato_unb.storage.db import init_db
from fato_unb.storage.repository import StagingRepository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("demo_pipeline")


async def run_demo():
    logger.info(
        "Iniciando demonstração do Pipeline Unificado (Ingestão -> Staging -> Chunks -> Embeddings -> Qdrant)..."
    )
    await init_db()
    await pipeline_job()

    repo = StagingRepository()
    stats = await repo.get_stats()
    logger.info(f"Estatísticas finais do Staging: {stats}")


if __name__ == "__main__":
    asyncio.run(run_demo())