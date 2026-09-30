import os
from functools import cache

from fastembed import SparseTextEmbedding, TextEmbedding

DEFAULT_DENSE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"

# O fastembed NÃO aplica estes prefixos (verificado no código do fastembed 0.8): sem eles o e5
# recupera pior sem nenhum erro. O jina-embeddings-v3 não precisa: usa task id (query/passage).
_PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/multilingual-e5": ("query: ", "passage: "),
    "intfloat/e5": ("query: ", "passage: "),
}


def model_prefixes(model_name: str) -> tuple[str, str]:
    """(prefixo de query, prefixo de passagem) exigidos pelo modelo; vazios se não houver."""
    for prefix, pair in _PREFIXES.items():
        if model_name.startswith(prefix):
            return pair
    return "", ""


@cache
def model_dimension(model_name: str) -> int:
    """Dimensão do vetor denso, lida do catálogo do fastembed (não baixa nem carrega o modelo)."""
    for info in TextEmbedding.list_supported_models():
        if info["model"] == model_name:
            return int(info["dim"])
    raise ValueError(
        f"Modelo '{model_name}' não está no catálogo do fastembed; "
        "use um dos modelos de TextEmbedding.list_supported_models()."
    )


class EmbeddingService:
    def __init__(
        self,
        model_name: str | None = None,
        sparse_model_name: str | None = None,
        provider: str = "local",
        mock_dimension: int = 384,
    ):
        """
        - provider='local': fastembed (ONNX, CPU) para denso e BM25 para esparso.
        - provider='mock': vetores sintéticos para testes rápidos, sem baixar modelo.

        O modelo denso vem de `model_name`, senão da variável EMBEDDING_MODEL, senão do padrão.
        """
        self.provider = provider
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL") or DEFAULT_DENSE_MODEL
        self.sparse_model_name = (
            sparse_model_name or os.getenv("SPARSE_MODEL") or DEFAULT_SPARSE_MODEL
        )
        self.mock_dimension = mock_dimension
        self.query_prefix, self.passage_prefix = model_prefixes(self.model_name)
        self._model: TextEmbedding | None = None
        self._sparse_model: SparseTextEmbedding | None = None

        if provider not in ("local", "mock"):
            raise ValueError(
                f"Provedor de embedding '{provider}' não suportado. Use 'local' ou 'mock'."
            )
        if self.provider == "local":
            model_dimension(self.model_name)  # falha cedo se o modelo não existir no catálogo
            self._model = TextEmbedding(model_name=self.model_name)
            self._sparse_model = SparseTextEmbedding(model_name=self.sparse_model_name)

    @property
    def vector_dimension(self) -> int:
        """Dimensão do vetor denso (necessária para configurar a coleção no Qdrant)."""
        if self.provider == "mock":
            return self.mock_dimension
        return model_dimension(self.model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Gera vetores para uma lista de chunks ou documentos."""
        if not texts:
            return []

        if self.provider == "mock":
            return [[0.05] * self.mock_dimension for _ in texts]

        if self._model is None:
            raise RuntimeError("Modelo local de embeddings não foi inicializado.")

        prefixed = [f"{self.passage_prefix}{t}" for t in texts] if self.passage_prefix else texts
        return [vector.tolist() for vector in self._model.passage_embed(prefixed)]

    def embed_query(self, query: str) -> list[float]:
        """Gera vetor para a pergunta ou alegação do usuário."""
        if self.provider == "mock":
            return [0.05] * self.mock_dimension

        if self._model is None:
            raise RuntimeError("Modelo local de embeddings não foi inicializado.")

        query_generator = self._model.query_embed(f"{self.query_prefix}{query}")
        return next(iter(query_generator)).tolist()

    def embed_sparse_texts(self, texts: list[str]) -> list[dict[str, list]]:
        """Gera vetores esparsos (BM25) para uma lista de chunks ou documentos."""
        if not texts:
            return []

        if self.provider == "mock":
            return [{"indices": [1, 2], "values": [1.0, 0.5]} for _ in texts]

        if self._sparse_model is None:
            raise RuntimeError("Modelo local de embeddings esparsos não foi inicializado.")

        sparse_generator = self._sparse_model.embed(texts)
        results = []
        for vec in sparse_generator:
            results.append({
                "indices": vec.indices.tolist(),
                "values": vec.values.tolist(),
            })
        return results

    def embed_sparse_query(self, query: str) -> dict[str, list]:
        """Gera vetor esparso (BM25) para a pergunta ou alegação do usuário."""
        if self.provider == "mock":
            return {"indices": [1, 2], "values": [1.0, 0.5]}

        if self._sparse_model is None:
            raise RuntimeError("Modelo local de embeddings esparsos não foi inicializado.")

        sparse_generator = self._sparse_model.query_embed(query)
        vec = next(iter(sparse_generator))
        return {
            "indices": vec.indices.tolist(),
            "values": vec.values.tolist(),
        }

