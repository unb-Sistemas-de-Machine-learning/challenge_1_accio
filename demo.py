import asyncio
import logging
from src.fato_unb.ingestion.scheduler import pipeline_job

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("demo_script")

async def run_demo():
    logger.info("Iniciando a demonstração do Web Scraping...")
    
    await pipeline_job(output_file="dados.txt")
    
    logger.info("Demonstração concluída. Verifique o arquivo dados.txt e os logs acima.")

if __name__ == "__main__":
    asyncio.run(run_demo())