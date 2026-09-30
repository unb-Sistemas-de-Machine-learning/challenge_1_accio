"""Enriquece documentos de RSS que ficaram só com o resumo do feed (17 a 30 palavras).

Baixa o texto completo de cada um, atualiza o staging e devolve o documento para 'pending'
(o próximo ciclo do pipeline, ou `scripts/reindexar.py`, gera os chunks novos).
Por padrão só simula; use --aplicar para gravar.

    uv run python scripts/backfill_rss.py             # simulação
    uv run python scripts/backfill_rss.py --aplicar   # grava no staging
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fato_unb.ingestion.models import SourceType  # noqa: E402
from fato_unb.ingestion.rss import enrich_documents_with_full_text  # noqa: E402
from fato_unb.storage.db import init_db  # noqa: E402
from fato_unb.storage.repository import StagingRepository  # noqa: E402

logging.basicConfig(level=logging.WARNING)


async def main(aplicar: bool, max_palavras: int) -> None:
    await init_db()
    repo = StagingRepository()
    curtos = [
        d
        for d in await repo.get_documents_by_source_type(SourceType.RSS_NEWS)
        if len(d.content.split()) < max_palavras
    ]
    print(f"{len(curtos)} documentos de RSS com menos de {max_palavras} palavras.")
    if not curtos:
        return

    antes = {d.doc_id: len(d.content.split()) for d in curtos}
    await asyncio.to_thread(enrich_documents_with_full_text, curtos)

    atualizados = 0
    for d in curtos:
        depois = len(d.content.split())
        mudou = depois > antes[d.doc_id]
        print(f"  {antes[d.doc_id]:>4} -> {depois:>4} palavras {'OK ' if mudou else 'SEM MUDANÇA'} | {d.title[:60]}")
        if mudou and aplicar:
            await repo.update_content_and_requeue(d.doc_id, d.content)
            atualizados += 1

    if aplicar:
        print(f"\n{atualizados} documentos atualizados e devolvidos para 'pending'.")
    else:
        print("\nSimulação: nada foi gravado. Rode com --aplicar.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--aplicar", action="store_true", help="grava no staging")
    parser.add_argument("--max-palavras", type=int, default=60, help="só documentos abaixo deste tamanho")
    args = parser.parse_args()
    asyncio.run(main(args.aplicar, args.max_palavras))
