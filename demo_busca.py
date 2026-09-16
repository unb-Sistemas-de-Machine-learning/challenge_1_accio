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

def run_search(query: str, limit: int = 3):
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

    logger.info(f"Executando busca vetorial na coleção '{collection_name}'...")
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
        raw_text = payload.get("raw_text", "")

        print(f"[{i}] Relevância (Score): {score:.4f}")
        print(f"    Título:   {titulo}")
        print(f"    Fonte:    {fonte} (Ref: {semestre})")
        print(f"    URL:      {url}")
        print(f"    Trecho:   {raw_text[:250]}...")
        print("-" * 56)

def main():
    # Permite passar a busca pela linha de comando ou usar a padrão
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Como funciona o Restaurante Universitário (RU) da UnB?"

    run_search(query)

if __name__ == "__main__":
    main()
