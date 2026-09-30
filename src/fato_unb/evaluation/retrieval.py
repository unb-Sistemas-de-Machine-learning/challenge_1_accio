import time
from dataclasses import asdict, dataclass, field

from qdrant_client import QdrantClient

from fato_unb.evaluation.dataset import (
    CasoTeste,
    TipoCaso,
    contains_span,
    dedupe_corpus,
    expected_content_keys,
    normalize_url,
    url_to_content_keys,
)
from fato_unb.ingestion.models import RawDocument
from fato_unb.evaluation.legacy_chunker import LegacyWordChunker
from fato_unb.rag.chunker import SemanticChunker
from fato_unb.rag.embeddings import EmbeddingService
from fato_unb.rag.reranker import DEFAULT_RERANKER_MODEL, Reranker
from fato_unb.vectorstore.collections import ensure_collection
from fato_unb.vectorstore.operations import buscar, upsert_documents

KS = (1, 3, 5)


@dataclass(frozen=True)
class EvalConfig:
    name: str
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chunker: str = "legacy"  # "legacy" (janela de palavras, fase 0) ou "sentence"
    chunk_size: int = 400  # palavras
    overlap: int = 50  # legacy: palavras; sentence: frases
    sparse_on: str = "content"  # texto do BM25: "content" ou "raw_text"
    reranker: str | None = None  # modelo do cross-encoder; None = sem reranking
    candidatos: int = 20  # quantos candidatos da busca híbrida o reranker reordena
    sparse_idf: bool = False  # modificador IDF do Qdrant no índice esparso (False = comportamento da fase 0)
    dedupe: bool = True


def _sentence(name: str, **kw) -> EvalConfig:
    return EvalConfig(name=name, chunker="sentence", **{"chunk_size": 120, "overlap": 1, **kw})


E5 = "intfloat/multilingual-e5-large"
JINA3 = "jinaai/jina-embeddings-v3"
RERANKER = DEFAULT_RERANKER_MODEL

# Novas variantes (reranker) entram aqui na próxima fase.
CONFIGS: dict[str, EvalConfig] = {
    "baseline": EvalConfig(name="baseline"),
    "chunk80": _sentence("chunk80", chunk_size=80),
    "chunk120": _sentence("chunk120"),
    "chunk200": _sentence("chunk200", chunk_size=200),
    "chunk120-ov0": _sentence("chunk120-ov0", overlap=0),
    "chunk120-ov2": _sentence("chunk120-ov2", overlap=2),
    "chunk120-rawsparse": _sentence("chunk120-rawsparse", sparse_on="raw_text"),
    # Fase 2: IDF no BM25 e modelos de embedding (cada um com o chunker antigo e o novo)
    "baseline-idf": EvalConfig(name="baseline-idf", sparse_idf=True),
    "chunk120-idf": _sentence("chunk120-idf", sparse_idf=True),
    "baseline-idf-rr": EvalConfig(name="baseline-idf-rr", sparse_idf=True, reranker=RERANKER),
    "chunk120-idf-rr": _sentence("chunk120-idf-rr", sparse_idf=True, reranker=RERANKER),
    "e5-c400": EvalConfig(name="e5-c400", model_name=E5, sparse_idf=True),
    "e5-c120": _sentence("e5-c120", model_name=E5, sparse_idf=True),
    "jina3-c400": EvalConfig(name="jina3-c400", model_name=JINA3, sparse_idf=True),
    "jina3-c120": _sentence("jina3-c120", model_name=JINA3, sparse_idf=True),
    "e5-c120-rr": _sentence("e5-c120-rr", model_name=E5, sparse_idf=True, reranker=RERANKER),
    "e5-c400-rr": EvalConfig(name="e5-c400-rr", model_name=E5, sparse_idf=True, reranker=RERANKER),
}


def make_chunker(config: EvalConfig):
    if config.chunker == "legacy":
        return LegacyWordChunker(chunk_size=config.chunk_size, chunk_overlap=config.overlap)
    if config.chunker == "sentence":
        return SemanticChunker(chunk_size=config.chunk_size, overlap_sentences=config.overlap)
    raise ValueError(f"chunker desconhecido: {config.chunker}")


def recall_at_k(ranked_keys: list[str], relevantes: set[str], k: int) -> float:
    return 1.0 if any(key in relevantes for key in ranked_keys[:k]) else 0.0


def reciprocal_rank(ranked_keys: list[str], relevantes: set[str]) -> float:
    for pos, key in enumerate(ranked_keys, start=1):
        if key in relevantes:
            return 1.0 / pos
    return 0.0


@dataclass
class ResultadoCaso:
    id: str
    tipo: str
    alegacao: str
    desafio: str
    ranked_urls: list[str]
    top_score: float | None
    recall: dict[int, float] = field(default_factory=dict)
    rr: float | None = None
    # Por chunk: o trecho recuperado contém a evidência? (chunk = raw_text)
    chunk_recall: dict[int, float] = field(default_factory=dict)
    chunk_rr: float | None = None
    # Por contexto: o que o LLM receberia (parent_text, ou o próprio chunk se não houver)
    ctx_recall: dict[int, float] = field(default_factory=dict)
    ctx_words: float | None = None  # palavras de contexto nos 3 primeiros resultados


@dataclass
class ResultadoAvaliacao:
    config: EvalConfig
    n_docs: int
    n_chunks: int
    latencia_media_ms: float
    casos: list[ResultadoCaso]

    def agregado(
        self, tipos: set[str] | None = None, desafios: set[str] | None = None
    ) -> dict:
        """Métricas médias sobre os casos que têm evidência esperada."""
        avaliaveis = [
            c
            for c in self.casos
            if c.rr is not None
            and (tipos is None or c.tipo in tipos)
            and (desafios is None or c.desafio in desafios)
        ]
        if not avaliaveis:
            return {"n": 0}
        n = len(avaliaveis)
        out: dict = {"n": n, "mrr": sum(c.rr for c in avaliaveis) / n}
        for k in KS:
            out[f"recall@{k}"] = sum(c.recall[k] for c in avaliaveis) / n
        com_span = [c for c in avaliaveis if c.chunk_rr is not None]
        if com_span:
            m = len(com_span)
            out["chunk_mrr"] = sum(c.chunk_rr for c in com_span) / m
            out["ctx_words"] = sum(c.ctx_words for c in com_span) / m
            for k in KS:
                out[f"chunk@{k}"] = sum(c.chunk_recall[k] for c in com_span) / m
                out[f"ctx@{k}"] = sum(c.ctx_recall[k] for c in com_span) / m
        return out

    def por_tipo(self) -> dict[str, dict]:
        tipos = sorted({c.tipo for c in self.casos if c.rr is not None})
        return {t: self.agregado({t}) for t in tipos}

    def por_desafio(self) -> dict[str, dict]:
        desafios = sorted({c.desafio for c in self.casos if c.rr is not None})
        return {d: self.agregado(desafios={d}) for d in desafios}

    def to_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "n_docs": self.n_docs,
            "n_chunks": self.n_chunks,
            "latencia_media_ms": self.latencia_media_ms,
            "geral": self.agregado(),
            "por_tipo": self.por_tipo(),
            "por_desafio": self.por_desafio(),
            "casos": [asdict(c) for c in self.casos],
        }


def build_index(
    docs: list[RawDocument],
    config: EvalConfig,
    embedder: EmbeddingService,
    client: QdrantClient | None = None,
    collection: str = "eval_retrieval",
) -> tuple[QdrantClient, str, int]:
    """Indexa o corpus numa coleção isolada (em memória por padrão).

    Nunca toca na coleção de produção. Reaproveita `ensure_collection` e
    `upsert_documents` para medir exatamente o que o pipeline real produz.
    """
    client = client or QdrantClient(":memory:")
    chunker = make_chunker(config)
    ensure_collection(client, collection, embedder.vector_dimension, sparse_idf=config.sparse_idf)

    total = 0
    for doc in docs:
        chunks = chunker.chunk_document(doc)
        total += upsert_documents(
            embedder=embedder,
            chunks=chunks,
            client=client,
            collection_name=collection,
            sparse_on=config.sparse_on,
        )
    return client, collection, total


def avaliar(
    casos: list[CasoTeste],
    docs: list[RawDocument],
    config: EvalConfig,
    embedder: EmbeddingService | None = None,
    client: QdrantClient | None = None,
    collection: str = "eval_retrieval",
    reranker: Reranker | None = None,
) -> ResultadoAvaliacao:
    url_map = url_to_content_keys(docs)  # sempre sobre o corpus completo
    docs_indexados = dedupe_corpus(docs) if config.dedupe else docs
    embedder = embedder or EmbeddingService(provider="local", model_name=config.model_name)

    client, collection, n_chunks = build_index(docs_indexados, config, embedder, client, collection)

    resultados: list[ResultadoCaso] = []
    tempos: list[float] = []
    for caso in casos:
        t0 = time.perf_counter()
        resp = buscar(
            query=caso.alegacao,
            embedder=embedder,
            client=client,
            collection_name=collection,
            limit=max(KS),
            reranker=reranker,
            candidatos=config.candidatos,
        )
        tempos.append((time.perf_counter() - t0) * 1000)

        pontos = getattr(resp, "points", [])
        payloads = [p.payload or {} for p in pontos]
        ranked_urls = [(p.payload or {}).get("url", "") for p in pontos]
        ranked_keys = [url_map.get(normalize_url(u), "") for u in ranked_urls]

        res = ResultadoCaso(
            id=caso.id,
            tipo=caso.tipo.value,
            alegacao=caso.alegacao,
            desafio=caso.desafio,
            ranked_urls=ranked_urls,
            top_score=pontos[0].score if pontos else None,
        )
        if caso.tipo != TipoCaso.SEM_REGISTRO and caso.expected_urls:
            relevantes = expected_content_keys(caso, url_map)
            res.recall = {k: recall_at_k(ranked_keys, relevantes, k) for k in KS}
            res.rr = reciprocal_rank(ranked_keys, relevantes)
            if caso.evidence_spans:
                chunk_hits = [contains_span(pl.get("raw_text", ""), caso.evidence_spans) for pl in payloads]
                contextos = [pl.get("parent_text") or pl.get("raw_text", "") for pl in payloads]
                ctx_hits = [contains_span(t, caso.evidence_spans) for t in contextos]
                res.chunk_recall = {k: float(any(chunk_hits[:k])) for k in KS}
                res.ctx_recall = {k: float(any(ctx_hits[:k])) for k in KS}
                res.chunk_rr = next((1.0 / i for i, h in enumerate(chunk_hits, 1) if h), 0.0)
                res.ctx_words = float(sum(len(t.split()) for t in contextos[:3]))
        resultados.append(res)

    return ResultadoAvaliacao(
        config=config,
        n_docs=len(docs_indexados),
        n_chunks=n_chunks,
        latencia_media_ms=sum(tempos) / len(tempos) if tempos else 0.0,
        casos=resultados,
    )


def formatar_tabela(resultados: list[ResultadoAvaliacao]) -> str:
    cab = f"{'config':<14}{'docs':>5}{'chunks':>7}{'n':>4}" + "".join(
        f"{'R@' + str(k):>8}" for k in KS
    ) + f"{'MRR':>8}{'ms/q':>8}"
    linhas = [cab, "-" * len(cab)]
    for r in resultados:
        g = r.agregado()
        linhas.append(
            f"{r.config.name:<14}{r.n_docs:>5}{r.n_chunks:>7}{g['n']:>4}"
            + "".join(f"{g[f'recall@{k}']:>8.3f}" for k in KS)
            + f"{g['mrr']:>8.3f}{r.latencia_media_ms:>8.0f}"
        )
    return "\n".join(linhas)


def formatar_tabela_chunk(resultados: list[ResultadoAvaliacao]) -> str:
    """Métrica por chunk: o trecho recuperado (ou seu contexto) contém a evidência?"""
    cab = f"{'config':<20}{'n':>4}" + "".join(f"{'C@' + str(k):>7}" for k in KS)
    cab += f"{'C-MRR':>8}" + "".join(f"{'X@' + str(k):>7}" for k in KS) + f"{'ctx(p)':>8}"
    linhas = ["Por chunk (C) e por contexto entregue ao LLM (X); ctx(p) = palavras no top-3", cab, "-" * len(cab)]
    for r in resultados:
        g = r.agregado()
        if "chunk_mrr" not in g:
            continue
        linhas.append(
            f"{r.config.name:<20}{g['n']:>4}"
            + "".join(f"{g[f'chunk@{k}']:>7.3f}" for k in KS)
            + f"{g['chunk_mrr']:>8.3f}"
            + "".join(f"{g[f'ctx@{k}']:>7.3f}" for k in KS)
            + f"{g['ctx_words']:>8.0f}"
        )
    return "\n".join(linhas)


def _linha(nome: str, g: dict) -> str:
    base = (
        f"  {nome:<14} n={g['n']:<3} R@1={g['recall@1']:.2f} R@3={g['recall@3']:.2f} "
        f"R@5={g['recall@5']:.2f} MRR={g['mrr']:.2f}"
    )
    if "chunk@3" in g:
        base += f" | C@1={g['chunk@1']:.2f} C@3={g['chunk@3']:.2f} X@3={g['ctx@3']:.2f}"
    return base


def formatar_por_tipo(r: ResultadoAvaliacao) -> str:
    linhas = [f"[{r.config.name}] por tipo"]
    linhas += [_linha(t, g) for t, g in r.por_tipo().items()]
    linhas.append(f"[{r.config.name}] por desafio")
    linhas += [_linha(d, g) for d, g in r.por_desafio().items()]
    return "\n".join(linhas)
