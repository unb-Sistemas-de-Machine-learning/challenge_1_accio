import sys
import logging
from pathlib import Path

# Garante importação do pacote fato_unb a partir de src
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fato_unb.rag.retriever import Retriever
from fato_unb.vectorstore.client import get_qdrant_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("demo_busca")

def run_search(query: str, limit: int = 3, completo: bool = False):
    client = get_qdrant_client()
    logger.info("Carregando modelos (embedding e, se configurado, reranker)...")
    retriever = Retriever.from_env()
    collection_name = retriever.collection_name

    if not client.collection_exists(collection_name):
        logger.error(
            f"Coleção '{collection_name}' não encontrada no Qdrant.\n"
            "Suba os containers (`docker compose up -d`) e indexe: `uv run python demo.py` "
            "ou `uv run python scripts/reindexar.py`."
        )
        return

    logger.info(f"Consultando '{collection_name}': '{query}'")
    evidencias = retriever.buscar(query, limit=limit)
    if not evidencias:
        print("\nNenhum documento relevante encontrado para essa consulta.")
        return

    modo = "reranker" if retriever.reranker else "busca híbrida (RRF)"
    print("\n========================================================")
    print(f" EVIDÊNCIAS PARA: '{query}'")
    print(f" Modelo: {retriever.embedder.model_name} | Ordenação: {modo}")
    print(f" Páginas distintas retornadas: {len(evidencias)}")
    print("========================================================\n")

    for i, ev in enumerate(evidencias, 1):
        print(f"[{i}] Score: {ev.score:.4f} ({'reranker' if ev.reranked else 'RRF'})")
        print(f"    Título:   {ev.title}")
        print(f"    Fonte:    {ev.source} (Ref: {ev.semester_ref or 'Geral'}) | {ev.published_at[:10]}")
        print(f"    URL:      {ev.url}")
        print("\n    --- TRECHO (chunk recuperado) ---")
        print(f"{ev.trecho.strip()}\n")
        if completo:
            print("    --- CONTEXTO ENTREGUE AO LLM (parent_text) ---")
            print(f"{ev.contexto.strip()}\n")
        else:
            print("    (Dica: use -c ou --completo para ver também o contexto maior entregue ao LLM)")
        print("-" * 56)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Busca híbrida de notícias no Qdrant.")
    parser.add_argument(
        "query",
        nargs="*",
        default=["Como funciona o Restaurante Universitário (RU) da UnB?"],
        help="Pergunta ou termos de busca",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=3,
        help="Número máximo de resultados (padrão: 3)",
    )
    parser.add_argument(
        "-c",
        "--completo",
        action="store_true",
        help="Exibe o trecho completo armazenado no banco sem truncamento",
    )

    args = parser.parse_args()
    query_str = " ".join(args.query) if isinstance(args.query, list) else args.query
    run_search(query=query_str, limit=args.limit, completo=args.completo)


if __name__ == "__main__":
    main()
