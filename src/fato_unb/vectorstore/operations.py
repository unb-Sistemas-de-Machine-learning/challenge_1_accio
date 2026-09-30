import logging
import re
import uuid
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseVector,
)

from fato_unb.rag.embeddings import EmbeddingService
from fato_unb.rag.models import DocumentChunk
from fato_unb.vectorstore.client import get_qdrant_client

logger = logging.getLogger(__name__)
NAMESPACE = uuid.UUID("f77e4b30-9222-4d60-889a-861c4da36012")

UNB_ACRONYMS = {
    r"\bru\b": "RU restaurante universitário",
    r"\bpas\b": "PAS programa de avaliação seriada",
    r"\bsigaa\b": "SIGAA sistema integrado de gestão de atividades acadêmicas",
    r"\bicc\b": "ICC instituto central de ciências",
    r"\bcepe\b": "CEPE conselho de ensino pesquisa e extensão",
    r"\bdeg\b": "DEG decanato de ensino de graduação",
    r"\bdac\b": "DAC decanato de assuntos comunitários",
}


def expand_acronyms(text: str) -> str:
    """Expande siglas comuns da UnB para melhorar a recuperação léxica e semântica."""
    expanded = text
    for pattern, replacement in UNB_ACRONYMS.items():
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    return expanded


def upsert_documents(
    embedder: EmbeddingService,
    chunks: list[DocumentChunk],
    client: QdrantClient | None = None,
    collection_name: str = "fato_unb_noticias",
    wait: bool = True,
    sparse_on: str = "content",
) -> int:
    """Calcula embeddings densos e esparsos (BM25) e persiste chunks no Qdrant de forma idempotente.

    O denso usa sempre `content` (com cabeçalho de contexto). `sparse_on` escolhe o texto do BM25:
    "content" (com cabeçalho) ou "raw_text" (só o trecho, sem repetir título e fonte em todo chunk).
    """
    if sparse_on not in ("content", "raw_text"):
        raise ValueError("sparse_on deve ser 'content' ou 'raw_text'")
    if not chunks:
        return 0

    qdrant = client or get_qdrant_client()
    textos = [chunk.content for chunk in chunks]
    dense_vectors = embedder.embed_texts(textos)
    sparse_dicts = embedder.embed_sparse_texts(
        [getattr(chunk, sparse_on) for chunk in chunks]
    )

    pontos = []
    for chunk, d_vec, s_dict in zip(chunks, dense_vectors, sparse_dicts):
        vectors = {
            "dense": d_vec,
            "sparse": SparseVector(
                indices=s_dict["indices"],
                values=s_dict["values"],
            ),
        }
        pontos.append(
            PointStruct(
                id=str(uuid.uuid5(NAMESPACE, chunk.chunk_id)),
                vector=vectors,
                payload=chunk.model_dump(mode="json"),
            )
        )

    qdrant.upsert(collection_name=collection_name, points=pontos, wait=wait)
    return len(pontos)


def buscar(
    query: str,
    embedder: EmbeddingService,
    client: QdrantClient | None = None,
    collection_name: str = "fato_unb_noticias",
    source: str | None = None,
    semester_ref: str | None = None,
    data_inicio: datetime | None = None,
    data_fim: datetime | None = None,
    limit: int = 5,
):
    qdrant = client or get_qdrant_client()
    query_expanded = expand_acronyms(query)

    vetor_denso = embedder.embed_query(query_expanded)
    vetor_esparso_dict = embedder.embed_sparse_query(query_expanded)
    vetor_esparso = SparseVector(
        indices=vetor_esparso_dict["indices"],
        values=vetor_esparso_dict["values"],
    )

    condicoes = []
    if source is not None:
        condicoes.append(FieldCondition(key="source", match=MatchValue(value=source)))
    if semester_ref is not None:
        condicoes.append(FieldCondition(key="semester_ref", match=MatchValue(value=semester_ref)))
    if data_inicio or data_fim:
        condicoes.append(FieldCondition(key="published_at", range=DatetimeRange(gte=data_inicio, lte=data_fim)))

    filtro = Filter(must=condicoes) if condicoes else None

    try:
        # Busca híbrida nativa com Reciprocal Rank Fusion (RRF)
        return qdrant.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(query=vetor_denso, using="dense", limit=limit * 2, filter=filtro),
                Prefetch(query=vetor_esparso, using="sparse", limit=limit * 2, filter=filtro),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
        )
    except Exception as exc:
        logger.warning(
            f"Busca híbrida indisponível na coleção '{collection_name}', recorrendo à busca densa: {exc}"
        )
        return qdrant.query_points(
            collection_name=collection_name,
            query=vetor_denso,
            using="dense",
            query_filter=filtro,
            limit=limit,
        )