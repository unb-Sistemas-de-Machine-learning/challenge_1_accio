"""Avalia a qualidade da recuperação (recall@k e MRR) para uma ou mais configurações.

Uso:
    uv run python scripts/avaliar.py --config baseline
    uv run python scripts/avaliar.py --config baseline --raw     # sem deduplicar o corpus
    uv run python scripts/avaliar.py --config baseline --json saida.json
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fato_unb.evaluation.dataset import (  # noqa: E402
    DEFAULT_CORPUS,
    DEFAULT_DATASET,
    load_corpus,
    load_dataset,
)
from fato_unb.rag.embeddings import EmbeddingService  # noqa: E402
from fato_unb.rag.reranker import Reranker  # noqa: E402
from fato_unb.evaluation.retrieval import (  # noqa: E402
    CONFIGS,
    avaliar,
    formatar_por_tipo,
    formatar_tabela,
    formatar_tabela_chunk,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", nargs="+", default=["baseline"], choices=sorted(CONFIGS))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--raw", action="store_true", help="indexa o corpus sem deduplicar conteúdo idêntico")
    parser.add_argument("--falhas", action="store_true", help="lista os casos em que a evidência não veio no top-5")
    parser.add_argument("--json", help="salva o resultado completo neste arquivo")
    args = parser.parse_args()

    casos = load_dataset(args.dataset)
    docs = load_corpus(args.corpus)

    resultados = []
    rerankers: dict[str, Reranker] = {}
    embedders: dict[str, EmbeddingService] = {}  # modelo atual, reaproveitado entre configs
    for nome in args.config:
        cfg = CONFIGS[nome]
        if args.raw:
            cfg = replace(cfg, name=f"{nome}-raw", dedupe=False)
        if cfg.model_name not in embedders:
            embedders.clear()  # só um modelo grande por vez na memória (e5-large ~2,2 GB, jina-v3 ~2,3 GB)
            gc.collect()
            embedders[cfg.model_name] = EmbeddingService(provider="local", model_name=cfg.model_name)
        embedder = embedders[cfg.model_name]
        reranker = None
        if cfg.reranker:
            if cfg.reranker not in rerankers:
                rerankers[cfg.reranker] = Reranker(model_name=cfg.reranker)
            reranker = rerankers[cfg.reranker]
        t0 = time.perf_counter()
        resultados.append(avaliar(casos, docs, cfg, embedder=embedder, reranker=reranker))
        print(
            f"[{len(resultados)}/{len(args.config)}] {cfg.name} pronto em {time.perf_counter() - t0:.0f}s",
            file=sys.stderr,
            flush=True,
        )

    print(formatar_tabela(resultados))
    print()
    print(formatar_tabela_chunk(resultados))
    for r in resultados:
        print()
        print(formatar_por_tipo(r))
        if args.falhas:
            print(f"[{r.config.name}] casos sem evidência no top-5:")
            for c in r.casos:
                if c.rr == 0.0:
                    print(f"  {c.id} ({c.tipo}) {c.alegacao}")

    if args.json:
        Path(args.json).write_text(
            json.dumps([r.to_dict() for r in resultados], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
