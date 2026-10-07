"""Reindexa todos os documentos do staging com o chunker/modelo atuais.

Use depois de trocar EMBEDDING_MODEL, o chunker ou a configuração do índice esparso.
A coleção do modelo em uso é recriada do zero (a de outros modelos não é tocada).

    uv run python scripts/reindexar.py            # pede confirmação
    uv run python scripts/reindexar.py --sim      # sem pergunta
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fato_unb.rag.embeddings import EmbeddingService  # noqa: E402
from fato_unb.rag.pipeline import IndexingPipeline  # noqa: E402
from fato_unb.storage.db import init_db  # noqa: E402
from fato_unb.storage.repository import StagingRepository  # noqa: E402
from fato_unb.vectorstore.client import get_qdrant_client  # noqa: E402
from fato_unb.vectorstore.collections import collection_name_for  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reindexar")


async def main(sim: bool) -> None:
    embedder = EmbeddingService(provider="local")
    colecao = collection_name_for(embedder)
    client = get_qdrant_client()

    if not sim:
        resp = input(
            f"Isto APAGA e recria a coleção '{colecao}' (modelo {embedder.model_name}) e reindexa "
            "todo o staging. Continuar? [s/N] "
        )
        if resp.strip().lower() not in ("s", "sim", "y", "yes"):
            print("Cancelado.")
            return

    await init_db()
    repo = StagingRepository()
    if client.collection_exists(colecao):
        client.delete_collection(colecao)
        logger.info(f"Coleção '{colecao}' removida.")
    resetados = await repo.reset_all_to_pending()
    logger.info(f"{resetados} documentos voltaram para 'pending'.")

    pipeline = IndexingPipeline(repository=repo, embedder=embedder, qdrant_client=client, collection_name=colecao)
    relatorio = await pipeline.run(max_docs=max(resetados, 1))
    logger.info(f"Reindexação concluída: {relatorio}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--sim", action="store_true", help="não pedir confirmação")
    asyncio.run(main(parser.parse_args().sim))
