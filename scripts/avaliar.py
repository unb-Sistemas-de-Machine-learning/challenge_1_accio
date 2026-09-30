"""Avalia a qualidade da recuperação (recall@k e MRR) para uma ou mais configurações.

Uso:
    uv run python scripts/avaliar.py --config baseline
    uv run python scripts/avaliar.py --config baseline --raw     # sem deduplicar o corpus
    uv run python scripts/avaliar.py --config baseline --json saida.json
"""

import argparse
import json
import sys
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
from fato_unb.evaluation.retrieval import (  # noqa: E402
    CONFIGS,
    avaliar,
    formatar_por_tipo,
    formatar_tabela,
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
    embedders: dict[str, EmbeddingService] = {}  # um modelo carregado por nome, reaproveitado
    for nome in args.config:
        cfg = CONFIGS[nome]
        if args.raw:
            cfg = replace(cfg, name=f"{nome}-raw", dedupe=False)
        if cfg.model_name not in embedders:
            embedders[cfg.model_name] = EmbeddingService(provider="local", model_name=cfg.model_name)
        embedder = embedders[cfg.model_name]
        resultados.append(avaliar(casos, docs, cfg, embedder=embedder))

    print(formatar_tabela(resultados))
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
