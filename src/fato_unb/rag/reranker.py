import os

from fastembed.rerank.cross_encoder import TextCrossEncoder

DEFAULT_RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"


class Reranker:
    """Cross-encoder que reordena os candidatos da busca híbrida.

    O cross-encoder lê a alegação e o trecho juntos, então separa melhor o trecho que responde
    do que apenas parece com a pergunta. Custa mais que o bi-encoder, por isso só recebe os
    ~20 candidatos que a busca híbrida já trouxe.

    - provider='local': fastembed (ONNX, CPU).
    - provider='mock': pontuação sintética (sobreposição de palavras) para testes, sem modelo.
    """

    def __init__(self, model_name: str | None = None, provider: str = "local"):
        if provider not in ("local", "mock"):
            raise ValueError(f"Provedor de reranker '{provider}' não suportado. Use 'local' ou 'mock'.")
        self.provider = provider
        self.model_name = model_name or os.getenv("RERANKER_MODEL") or DEFAULT_RERANKER_MODEL
        self._model: TextCrossEncoder | None = None
        if provider == "local":
            self._model = TextCrossEncoder(model_name=self.model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if self.provider == "mock":
            q = set(query.lower().split())
            return [float(len(q & set(t.lower().split()))) for t in texts]
        assert self._model is not None
        return [float(s) for s in self._model.rerank(query, texts)]


def rerank_points(reranker: Reranker, query: str, points: list, top_k: int) -> list:
    """Reordena pontos do Qdrant pelo score do cross-encoder e devolve os `top_k` melhores.

    O `score` de cada ponto passa a ser o do reranker (comparável entre consultas, ao contrário
    do score de RRF, que só reflete posição); o score original fica em `payload["_score_busca"]`.
    """
    if not points:
        return []
    textos = [(p.payload or {}).get("content") or (p.payload or {}).get("raw_text", "") for p in points]
    scores = reranker.score(query, textos)
    ordenados = sorted(zip(points, scores), key=lambda x: x[1], reverse=True)[:top_k]
    resultado = []
    for ponto, score in ordenados:
        payload = dict(ponto.payload or {})
        payload["_score_busca"] = ponto.score
        resultado.append(ponto.model_copy(update={"score": score, "payload": payload}))
    return resultado
