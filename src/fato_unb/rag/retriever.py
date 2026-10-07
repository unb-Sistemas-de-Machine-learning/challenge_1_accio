import os
from dataclasses import dataclass
from datetime import datetime

from qdrant_client import QdrantClient

from fato_unb.rag.embeddings import EmbeddingService
from fato_unb.rag.reranker import Reranker
from fato_unb.vectorstore.client import get_qdrant_client
from fato_unb.vectorstore.collections import collection_name_for
from fato_unb.vectorstore.operations import buscar


@dataclass(frozen=True)
class Evidencia:
    """Uma fonte pronta para o LLM/bot citar."""

    doc_id: str
    title: str
    url: str
    source: str
    published_at: str
    semester_ref: str | None
    trecho: str  # o chunk que casou com a alegação
    contexto: str  # o que entregar ao LLM: parent_text (trecho ao redor) ou o próprio chunk
    score: float  # do reranker quando há; senão do RRF (só reflete posição)
    reranked: bool


def _doc_key(url: str, trecho: str) -> tuple[str, str]:
    """Chave para descartar cópias: a mesma página com '#main' ou barra final, ou o mesmo texto."""
    return url.split("#", 1)[0].rstrip("/"), " ".join(trecho.split())


class Retriever:
    """Busca evidências para uma alegação: híbrida (denso + BM25), reranker opcional, sem cópias."""

    def __init__(
        self,
        embedder: EmbeddingService,
        client: QdrantClient | None = None,
        collection_name: str | None = None,
        reranker: Reranker | None = None,
        candidatos: int = 20,
    ):
        self.embedder = embedder
        self.client = client or get_qdrant_client()
        self.collection_name = collection_name or collection_name_for(embedder)
        self.reranker = reranker
        self.candidatos = candidatos

    @classmethod
    def from_env(cls) -> "Retriever":
        """Modelo de embedding via EMBEDDING_MODEL; reranker só se RERANKER_MODEL estiver definido."""
        reranker_model = os.getenv("RERANKER_MODEL")
        return cls(
            embedder=EmbeddingService(provider="local"),
            reranker=Reranker(model_name=reranker_model) if reranker_model else None,
        )

    def buscar(
        self,
        alegacao: str,
        limit: int = 5,
        source: str | None = None,
        semester_ref: str | None = None,
        data_inicio: datetime | None = None,
        data_fim: datetime | None = None,
    ) -> list[Evidencia]:
        """Até `limit` evidências de páginas distintas, da mais para a menos relevante."""
        # Um documento longo tem dezenas de chunks e pode ocupar todos os primeiros lugares.
        # Busca-se bem mais chunks do que `limit` para sobrar de páginas distintas após dedup.
        buscar_n = max(limit * 8, 30)
        resposta = buscar(
            query=alegacao,
            embedder=self.embedder,
            client=self.client,
            collection_name=self.collection_name,
            source=source,
            semester_ref=semester_ref,
            data_inicio=data_inicio,
            data_fim=data_fim,
            limit=buscar_n,
            reranker=self.reranker,
            candidatos=max(self.candidatos, buscar_n),
        )
        evidencias: list[Evidencia] = []
        vistos_url: set[str] = set()
        vistos_texto: set[str] = set()
        for ponto in getattr(resposta, "points", []):
            pl = ponto.payload or {}
            url_key, texto_key = _doc_key(str(pl.get("url", "")), pl.get("raw_text", ""))
            if url_key in vistos_url or texto_key in vistos_texto:
                continue
            vistos_url.add(url_key)
            vistos_texto.add(texto_key)
            evidencias.append(
                Evidencia(
                    doc_id=pl.get("doc_id", ""),
                    title=pl.get("title", ""),
                    url=str(pl.get("url", "")),
                    source=pl.get("source", ""),
                    published_at=str(pl.get("published_at", "")),
                    semester_ref=pl.get("semester_ref"),
                    trecho=pl.get("raw_text", ""),
                    contexto=pl.get("parent_text") or pl.get("raw_text", ""),
                    score=float(ponto.score),
                    reranked=self.reranker is not None,
                )
            )
            if len(evidencias) == limit:
                break
        return evidencias
