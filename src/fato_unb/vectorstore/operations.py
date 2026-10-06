from fato_unb.rag.models import DocumentChunk
from fato_unb.vectorstore.client import get_qdrant_client
from fato_unb.rag.embeddings import EmbeddingService
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, FieldCondition, MatchValue, DatetimeRange, Filter
from datetime import datetime
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
    
def buscar(query: str, embedder: EmbeddingService, source: str | None = None, semester_ref: str | None = None, data_inicio: datetime | None = None, data_fim: datetime | None = None, limit: int = 5):
    client = get_qdrant_client()
    vetor = embedder.embed_query(query)
    
    condicoes = []
    if source is not None:
        condicoes.append(FieldCondition(key="source", match=MatchValue(value=source)))
    if semester_ref is not None:
        condicoes.append(FieldCondition(key="semester_ref", match=MatchValue(value=semester_ref)))
    if data_inicio or data_fim:
        condicoes.append(FieldCondition(key="published_at", range=DatetimeRange(gte=data_inicio, lte=data_fim)))
        
    filtro = Filter(must=condicoes) if condicoes else None
    return client.query_points(collection_name="fato_unb_noticias", query=vetor, using="dense", query_filter=filtro, limit=limit)