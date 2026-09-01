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