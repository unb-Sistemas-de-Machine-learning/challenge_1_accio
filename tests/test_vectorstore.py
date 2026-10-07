import pytest
from fato_unb.vectorstore.client import get_qdrant_client
from fato_unb.vectorstore.collections import create_collection
from fato_unb.rag.embeddings import EmbeddingService

TEST_COLLECTION_NAME = "test_fato_unb_noticias"


@pytest.fixture
def colecao_teste():
    client = get_qdrant_client()
    embedder = EmbeddingService(provider="mock")

    # TODO 1: chama create_collection passando o embedder E o nome de teste
    #         (usa o parâmetro novo que você acabou de criar)
    create_collection(embedder, TEST_COLLECTION_NAME)

    yield client  # o teste vai receber "client" como argumento

    # TODO 2: depois do yield, apaga TEST_COLLECTION_NAME
    #         (método do client, o oposto de create_collection — procura "delete" no README do qdrant-client)
    client.delete_collection(collection_name=TEST_COLLECTION_NAME)

def test_cria_colecao_se_nao_existir(colecao_teste):
    # colecao_teste aqui já é o "client" que o fixture entregou (o yield)
    # TODO 3: assert que colecao_teste.collection_exists(TEST_COLLECTION_NAME) é True
    assert colecao_teste.collection_exists(TEST_COLLECTION_NAME) == True

def test_nao_duplica_ao_chamar_duas_vezes(colecao_teste):
    embedder = EmbeddingService(provider="mock")
    # TODO 4: chama create_collection de novo, mesmo nome de teste
    #         se não levantar exceção, o teste já passa sozinho (não precisa de assert aqui)
    create_collection(embedder, TEST_COLLECTION_NAME)


def test_upsert_and_hybrid_search_in_memory():
    """Valida upsert de vetores denso/esparso e busca híbrida com RRF no Qdrant."""
    from datetime import UTC, datetime
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, SparseVectorParams, VectorParams
    from fato_unb.rag.models import DocumentChunk
    from fato_unb.vectorstore.operations import buscar, upsert_documents

    client = QdrantClient(":memory:")
    embedder = EmbeddingService(provider="mock", mock_dimension=384)
    collection = "test_hybrid"

    client.create_collection(
        collection_name=collection,
        vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )

    chunk = DocumentChunk(
        chunk_id="chunk1234567890a",
        doc_id="doc1234567890ab",
        chunk_index=0,
        total_chunks=1,
        content="[Documento: RU] Horário de funcionamento do Restaurante Universitário.",
        raw_text="Horário de funcionamento do Restaurante Universitário.",
        title="Funcionamento do RU",
        url="https://noticias.unb.br/ru",
        source="UnB Notícias",
        published_at=datetime.now(UTC),
    )

    count = upsert_documents(embedder=embedder, chunks=[chunk], client=client, collection_name=collection)
    assert count == 1

    # Executa busca híbrida buscando por 'RU' (que expande para 'RU restaurante universitário')
    res = buscar(embedder=embedder, query="RU", client=client, collection_name=collection, limit=2)
    assert len(res.points) == 1
    assert res.points[0].payload["title"] == "Funcionamento do RU"


def test_buscar_fallback_dense():
    """Valida que buscar faz fallback limpo para busca densa se a coleção não suportar sparse."""
    from datetime import UTC, datetime
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from fato_unb.vectorstore.operations import buscar

    client = QdrantClient(":memory:")
    embedder = EmbeddingService(provider="mock", mock_dimension=384)
    collection = "test_dense_only"

    client.create_collection(
        collection_name=collection,
        vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
    )

    # Insere apenas com dense vector
    dense_vec = embedder.embed_query("teste")
    client.upsert(
        collection_name=collection,
        points=[
            PointStruct(
                id="c0a80101-0000-0000-0000-000000000001",
                vector={"dense": dense_vec},
                payload={"title": "Notícia Apenas Densa"},
            )
        ],
    )

    # Deve executar fallback sem estourar erro
    res = buscar(embedder=embedder, query="teste", client=client, collection_name=collection, limit=1)
    assert len(res.points) == 1
    assert res.points[0].payload["title"] == "Notícia Apenas Densa"