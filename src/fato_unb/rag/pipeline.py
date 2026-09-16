import logging
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, SparseVectorParams, VectorParams

from fato_unb.rag.chunker import SemanticChunker
from fato_unb.rag.embeddings import EmbeddingService
from fato_unb.storage.repository import StagingRepository
from fato_unb.vectorstore.client import get_qdrant_client

logger = logging.getLogger(__name__)
NAMESPACE = uuid.UUID("f77e4b30-9222-4d60-889a-861c4da36012")


@dataclass
class IndexingReport:
    total_processed: int = 0
    total_indexed: int = 0
    total_chunks: int = 0
    total_failed: int = 0


class IndexingPipeline:
    def __init__(
        self,
        repository: StagingRepository | None = None,
        chunker: SemanticChunker | None = None,
        embedder: EmbeddingService | None = None,
        qdrant_client: QdrantClient | None = None,
        collection_name: str = "fato_unb_noticias",
        batch_size: int = 64,
    ):
        self.repository = repository or StagingRepository()
        self.chunker = chunker or SemanticChunker()
        self.embedder = embedder or EmbeddingService(provider="local")
        self.client = qdrant_client or get_qdrant_client()
        self.collection_name = collection_name
        self.batch_size = batch_size

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.embedder.vector_dimension,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={"sparse": SparseVectorParams()},
            )
            logger.info(f"Coleção Qdrant '{self.collection_name}' criada com sucesso.")

    async def run(self, max_docs: int = 100) -> IndexingReport:
        report = IndexingReport()
        self.ensure_collection()

        docs = await self.repository.get_pending_documents(limit=max_docs)
        if not docs:
            logger.info("Nenhum documento pendente para indexação.")
            return report

        logger.info(f"Iniciando pipeline de indexação para {len(docs)} documentos pendentes.")

        for doc in docs:
            report.total_processed += 1
            try:
                chunks = self.chunker.chunk_document(doc)
                if not chunks:
                    logger.warning(f"Documento {doc.doc_id} ({doc.title}) não gerou chunks.")
                    await self.repository.mark_as_failed(
                        doc.doc_id, "Corpo de texto vazio ou não particionável"
                    )
                    report.total_failed += 1
                    continue

                # Processa chunks em lotes de embedding
                for i in range(0, len(chunks), self.batch_size):
                    chunk_batch = chunks[i : i + self.batch_size]
                    texts = [c.content for c in chunk_batch]
                    vectors = self.embedder.embed_texts(texts)

                    points = [
                        PointStruct(
                            id=str(uuid.uuid5(NAMESPACE, chunk.chunk_id)),
                            vector={"dense": vec},
                            payload=chunk.model_dump(mode="json"),
                        )
                        for chunk, vec in zip(chunk_batch, vectors)
                    ]

                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                        wait=True,
                    )
                    report.total_chunks += len(points)

                await self.repository.mark_as_indexed(doc.doc_id)
                report.total_indexed += 1

            except Exception as exc:
                logger.error(f"Erro ao indexar documento {doc.doc_id}: {exc}", exc_info=True)
                await self.repository.mark_as_failed(doc.doc_id, str(exc))
                report.total_failed += 1

        logger.info(f"Indexação finalizada: {report}")
        return report
