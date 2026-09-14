from fato_unb.rag.models import DocumentChunk
from fato_unb.vectorstore.client import get_qdrant_client
from fato_unb.rag.embeddings import EmbeddingService
from qdrant_client.models import PointStruct
import uuid
NAMESPACE = uuid.UUID("f77e4b30-9222-4d60-889a-861c4da36012")

def upsert_documents(embedder: EmbeddingService, chunks: list[DocumentChunk]) -> None:
    client = get_qdrant_client()
    textos = [chunk.content for chunk in chunks]
    vetores = embedder.embed_texts(textos)
    
    pontos = [
        PointStruct(
            id=str(uuid.uuid5(NAMESPACE, chunk.chunk_id)),
            vector={"dense": vetor},
            payload=chunk.model_dump(mode="json"),
        )
        for chunk, vetor in zip(chunks, vetores)
    ]

    client.upsert(collection_name="fato_unb_noticias", points=pontos)