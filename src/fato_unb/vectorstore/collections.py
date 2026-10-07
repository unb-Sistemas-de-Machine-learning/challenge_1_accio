import logging
import re

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier, SparseVectorParams, VectorParams

from fato_unb.rag.embeddings import LEGACY_DENSE_MODEL, EmbeddingService
from fato_unb.vectorstore.client import get_qdrant_client

logger = logging.getLogger(__name__)

BASE_COLLECTION = "fato_unb_noticias"


def collection_name_for(embedder: EmbeddingService, base: str = BASE_COLLECTION) -> str:
    """Nome da coleção para o modelo denso em uso.

    Só o MiniLM, usado nas primeiras fases, mantém o nome histórico; os demais modelos (inclusive
    o padrão atual) ganham sufixo, porque vetores de modelos diferentes (dimensão e espaço) não
    podem conviver na mesma coleção.
    """
    if embedder.model_name == LEGACY_DENSE_MODEL:
        return base
    slug = re.sub(r"[^a-z0-9]+", "-", embedder.model_name.split("/")[-1].lower()).strip("-")
    return f"{base}__{slug}"


def _dense_size(client: QdrantClient, name: str) -> int | None:
    vectors = client.get_collection(name).config.params.vectors
    dense = vectors.get("dense") if isinstance(vectors, dict) else None
    return dense.size if dense is not None else None


def ensure_collection(
    client: QdrantClient,
    name: str,
    vector_size: int,
    sparse_idf: bool = True,
) -> bool:
    """Cria a coleção híbrida (denso + esparso) se não existir; devolve True se criou.

    - `sparse_idf`: o BM25 do fastembed só calcula IDF se o Qdrant aplicar o modificador IDF.
      Sem ele os termos raros (ex.: nomes próprios) não pesam mais que palavras comuns.
    - Se a coleção já existe com outra dimensão, levanta erro em vez de falhar só no upsert.
    """
    if client.collection_exists(name):
        existing = _dense_size(client, name)
        if existing is not None and existing != vector_size:
            raise ValueError(
                f"A coleção '{name}' tem vetores densos de dimensão {existing}, mas o modelo "
                f"atual gera {vector_size}. Use outra coleção (collection_name_for) ou reindexe."
            )
        return False

    client.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=vector_size, distance=Distance.COSINE)},
        sparse_vectors_config={
            "sparse": SparseVectorParams(modifier=Modifier.IDF if sparse_idf else None)
        },
    )
    logger.info(f"Coleção Qdrant '{name}' criada (denso={vector_size}, idf={sparse_idf}).")
    return True


def create_collection(embedder: EmbeddingService, collection: str | None = None) -> None:
    """Compatibilidade: cria a coleção do modelo em uso no Qdrant configurado no ambiente."""
    name = collection or collection_name_for(embedder)
    if not ensure_collection(get_qdrant_client(), name, embedder.vector_dimension):
        print(f"Collection nomeada como '{name}' já criada.")
