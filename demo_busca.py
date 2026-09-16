import sys
import logging
from pathlib import Path

# Garante importação do pacote fato_unb a partir de src
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fato_unb.rag.embeddings import EmbeddingService
from fato_unb.vectorstore.operations import buscar
from fato_unb.vectorstore.client import get_qdrant_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("demo_busca")

def run_search(query: str, limit: int = 3, completo: bool = False):
    client = get_qdrant_client()
    collection_name = "fato_unb_noticias"

    if not client.collection_exists(collection_name):
        logger.error(
            f"Coleção '{collection_name}' não encontrada no Qdrant.\n"
            "Certifique-se de iniciar os containers (`docker compose up -d`) e rodar o pipeline primeiro (`uv run python demo.py`)."
        )
        return

    logger.info(f"Carregando modelo de embeddings para consultar: '{query}'...")
    embedder = EmbeddingService(provider="local")

    logger.info(f"Executando busca vetorial híbrida na coleção '{collection_name}'...")
    resultado = buscar(query=query, embedder=embedder, limit=limit)

    pontos = getattr(resultado, "points", [])
    if not pontos:
        print("\nNenhum documento relevante encontrado para essa consulta.")
        return

    print(f"\n========================================================")
    print(f" RESULTADOS DA BUSCA SEMÂNTICA NO QDRANT")
    print(f" Pergunta: '{query}'")
    print(f" Total de chunks retornados: {len(pontos)}")
    print(f"========================================================\n")

    for i, ponto in enumerate(pontos, 1):
        payload = ponto.payload or {}
        score = getattr(ponto, "score", 0.0)
        titulo = payload.get("title", "Sem título")
        fonte = payload.get("source", "Desconhecida")
        semestre = payload.get("semester_ref", "Geral")
        url = payload.get("url", "Sem URL")
        chunk_idx = payload.get("chunk_index", 0)
        total_chunks = payload.get("total_chunks", 1)
        raw_text = payload.get("raw_text", "")

        print(f"[{i}] Relevância (Score): {score:.4f}")
        print(f"    Título:   {titulo}")
        print(f"    Fonte:    {fonte} (Ref: {semestre}) | Bloco: {chunk_idx + 1}/{total_chunks}")
        print(f"    URL:      {url}")

        if completo:
            print(f"\n    --- TRECHO COMPLETO DO BANCO (CHUNK) ---")
            print(f"{raw_text.strip()}\n")
        else:
            preview = raw_text[:250].strip() + ("..." if len(raw_text) > 250 else "")
            print(f"    Trecho:   {preview}")
            print("    (Dica: use -c ou --completo para ver o texto completo deste trecho)")
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
