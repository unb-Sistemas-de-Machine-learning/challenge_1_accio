from fato_unb.vectorstore.client import get_qdrant_client
from fato_unb.rag.embeddings import EmbeddingService
from qdrant_client.models import Distance, VectorParams, SparseVectorParams

def create_collection(embedder: EmbeddingService) -> None:
    client = get_qdrant_client()
    if client.collection_exists("fato_unb_noticias"):
        print("Collection nomeada como 'fato_unb_noticias' já criada.")
    else:
        client.create_collection(
            collection_name="fato_unb_noticias",
            vectors_config={"dense": VectorParams(size=embedder.vector_dimension, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()}
        )