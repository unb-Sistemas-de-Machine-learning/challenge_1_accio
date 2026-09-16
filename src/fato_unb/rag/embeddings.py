from fastembed import SparseTextEmbedding, TextEmbedding


class EmbeddingService:
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        sparse_model_name: str = "Qdrant/bm25",
        provider: str = "local",
        mock_dimension: int = 384,
    ):
        """
        - provider='local': Usa fastembed com ONNX Runtime (CPU ultra leve) para denso e BM25 para esparso.
        - provider='mock': Retorna vetores sintéticos para testes rápidos sem download de modelo.
        """
        self.provider = provider
        self.model_name = model_name
        self.sparse_model_name = sparse_model_name
        self.mock_dimension = mock_dimension
        self._model: TextEmbedding | None = None
        self._sparse_model: SparseTextEmbedding | None = None

        if provider not in ("local", "mock"):
            raise ValueError(
                f"Provedor de embedding '{provider}' não suportado. Use 'local' ou 'mock'."
            )
        if self.provider == "local":
            # Inicializa e carrega os modelos em memória
            self._model = TextEmbedding(model_name=self.model_name)
            self._sparse_model = SparseTextEmbedding(model_name=self.sparse_model_name)

        if self.provider == "mock":
            # Modo mock: não carrega nada em memória
            self._model = None
            self._sparse_model = None

    @property
    def vector_dimension(self) -> int:
        """Dimensão do vetor denso gerado (necessário para configurar a coleção no Qdrant)."""
        if self.provider == "mock":
            return self.mock_dimension
        # O paraphrase-multilingual-MiniLM-L12-v2 gera vetores de 384 dimensões
        return 384

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Gera vetores para uma lista de chunks ou documentos."""
        if not texts:
            return []

        if self.provider == "mock":
            return [[0.05] * self.mock_dimension for _ in texts]

        if self._model is None:
            raise RuntimeError("Modelo local de embeddings não foi inicializado.")

        # fastembed retorna um gerador de arrays numpy otimizados
        embeddings_generator = self._model.embed(texts)
        return [vector.tolist() for vector in embeddings_generator]

    def embed_query(self, query: str) -> list[float]:
        """Gera vetor para a pergunta ou alegação do usuário."""
        if self.provider == "mock":
            return [0.05] * self.mock_dimension

        if self._model is None:
            raise RuntimeError("Modelo local de embeddings não foi inicializado.")

        # fastembed possui otimização específica para queries de busca
        query_generator = self._model.query_embed(query)
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

