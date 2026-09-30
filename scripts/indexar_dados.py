import json
import logging

from fato_unb.ingestion.models import RawDocument
from fato_unb.rag.chunker import SemanticChunker
from fato_unb.rag.embeddings import EmbeddingService
from fato_unb.vectorstore.collections import create_collection
from fato_unb.vectorstore.operations import upsert_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def carregar_documentos(caminho: str) -> list[RawDocument]:
    documentos = []
    with open(caminho, encoding="utf-8") as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            if not linha:
                continue
            documentos.append(RawDocument(**json.loads(linha)))
    return documentos


def main() -> None:
    documentos = carregar_documentos("dados.txt")
    logger.info("Carregados %d documentos de dados.txt", len(documentos))

    embedder = EmbeddingService(provider="local")
    create_collection(embedder)

    chunker = SemanticChunker()
    total_chunks = 0
    for documento in documentos:
        chunks = chunker.chunk_document(documento)
        if not chunks:
            continue
        upsert_documents(embedder, chunks)
        total_chunks += len(chunks)

    logger.info("Indexação concluída: %d chunks salvos no Qdrant", total_chunks)


if __name__ == "__main__":
    main()
