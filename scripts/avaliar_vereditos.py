"""Mede o acerto dos vereditos do LLM no dataset (busca -> LLM -> guardrails).

    uv run python scripts/avaliar_vereditos.py                       # 81 casos rotulados, Gemini
    uv run python scripts/avaliar_vereditos.py --limite 10           # teste rápido e barato
    uv run python scripts/avaliar_vereditos.py --falhas              # lista cada erro com a justificativa
    uv run python scripts/avaliar_vereditos.py --provider anthropic --modelo claude-haiku-4-5

Cada resultado é gravado em --saida (JSONL); se o arquivo existir, a execução RETOMA de onde parou
e REFAZ os casos que falharam por indisponibilidade do provedor. --rpm respeita a cota (Gemini free: 15/min).
Use --saida diferente por modelo para comparar. A data de "hoje" é fixa (2026-09-30) porque os casos
`desatualizada` e `temporal` assumem esse dia; mude com --hoje.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fato_unb.evaluation.dataset import DEFAULT_CORPUS, DEFAULT_DATASET, load_corpus, load_dataset  # noqa: E402
from fato_unb.evaluation.verdicts import calcular_metricas, executar, formatar_relatorio  # noqa: E402
from fato_unb.llm.clients import LLMError, criar_cliente  # noqa: E402
from fato_unb.llm.guardrails import GuardrailConfig  # noqa: E402
from fato_unb.rag.retriever import Retriever  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--provider", choices=["gemini", "anthropic"])
    p.add_argument("--modelo")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--limite", type=int, help="avalia só os N primeiros casos")
    p.add_argument("--paralelismo", type=int, default=3, help="chamadas simultâneas ao LLM (padrão 3)")
    p.add_argument("--rpm", type=float, default=12, help="máx. requisições/minuto (padrão 12; free tier do Gemini: 15)")
    p.add_argument("--saida", default="vereditos.jsonl", help="arquivo JSONL de resultados (retomável)")
    p.add_argument("--hoje", default="2026-09-30", help="data tratada como 'hoje' (AAAA-MM-DD)")
    p.add_argument("--incluir-perguntas", action="store_true", help="roda também os casos sem rótulo")
    p.add_argument("--falhas", action="store_true", help="lista os erros")
    p.add_argument("--json", help="salva as métricas neste arquivo")
    a = p.parse_args()

    try:
        llm = criar_cliente(a.provider, a.modelo, rpm=a.rpm)
    except LLMError as exc:
        sys.exit(f"Erro: {exc}")

    casos = load_dataset(a.dataset)
    if a.limite:
        casos = [c for c in casos if c.veredito_esperado is not None][: a.limite]
    cfg = GuardrailConfig()
    resultados = executar(
        casos, load_corpus(a.corpus), Retriever.from_env(), llm,
        hoje=date.fromisoformat(a.hoje), paralelismo=a.paralelismo,
        saida=Path(a.saida), incluir_perguntas=a.incluir_perguntas, config=cfg,
    )
    # só os casos pedidos nesta execução entram no relatório (o arquivo pode ter mais)
    ids = {c.id for c in casos}
    resultados = [r for r in resultados if r.id in ids]
    m = calcular_metricas(resultados)
    print("\n" + formatar_relatorio(m, resultados, falhas=a.falhas))
    if a.json:
        Path(a.json).write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
