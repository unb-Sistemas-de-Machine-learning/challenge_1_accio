import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Modifier

from fato_unb.rag.embeddings import (
    DEFAULT_DENSE_MODEL,
    LEGACY_DENSE_MODEL,
    EmbeddingService,
    model_dimension,
    model_prefixes,
)
from fato_unb.vectorstore.collections import (
    BASE_COLLECTION,
    collection_name_for,
    ensure_collection,
)

E5 = "intfloat/multilingual-e5-large"


def test_model_prefixes_only_for_e5():
    assert model_prefixes(E5) == ("query: ", "passage: ")
    assert model_prefixes(LEGACY_DENSE_MODEL) == ("", "")
    assert model_prefixes("jinaai/jina-embeddings-v3") == ("", "")


def test_model_dimension_comes_from_catalog_without_loading_the_model():
    assert model_dimension(LEGACY_DENSE_MODEL) == 384
    assert model_dimension(E5) == 1024
    with pytest.raises(ValueError, match="catálogo"):
        model_dimension("modelo/que-nao-existe")


def test_default_model_is_e5_large():
    assert DEFAULT_DENSE_MODEL == E5


def test_embedding_model_can_come_from_environment(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", LEGACY_DENSE_MODEL)
    assert EmbeddingService(provider="mock").model_name == LEGACY_DENSE_MODEL
    monkeypatch.delenv("EMBEDDING_MODEL")
    assert EmbeddingService(provider="mock").model_name == DEFAULT_DENSE_MODEL


def test_explicit_model_name_wins_over_environment(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", E5)
    assert EmbeddingService(model_name=LEGACY_DENSE_MODEL, provider="mock").model_name == LEGACY_DENSE_MODEL


class _FakeModel:
    """Captura o texto que chegaria ao modelo, para verificar os prefixos sem baixar nada."""

    def __init__(self):
        self.passages: list[str] = []
        self.queries: list[str] = []

    def passage_embed(self, texts):
        self.passages += list(texts)
        return (np.zeros(3) for _ in texts)

    def query_embed(self, query):
        self.queries.append(query)
        return iter([np.zeros(3)])


def _service_with_fake_model(model_name: str) -> tuple[EmbeddingService, _FakeModel]:
    service = EmbeddingService(model_name=model_name, provider="mock")
    fake = _FakeModel()
    service.provider, service._model = "local", fake
    return service, fake


def test_e5_receives_query_and_passage_prefixes():
    service, fake = _service_with_fake_model(E5)
    service.embed_texts(["texto A", "texto B"])
    service.embed_query("pergunta")
    assert fake.passages == ["passage: texto A", "passage: texto B"]
    assert fake.queries == ["query: pergunta"]


def test_models_without_prefix_receive_raw_text():
    service, fake = _service_with_fake_model(LEGACY_DENSE_MODEL)
    service.embed_texts(["texto A"])
    service.embed_query("pergunta")
    assert fake.passages == ["texto A"]
    assert fake.queries == ["pergunta"]


def test_collection_name_keeps_legacy_name_only_for_the_minilm():
    assert collection_name_for(EmbeddingService(model_name=LEGACY_DENSE_MODEL, provider="mock")) == BASE_COLLECTION
    # o modelo padrão atual (e5-large) usa coleção própria, que não colide com os dados do MiniLM
    assert collection_name_for(EmbeddingService(provider="mock")) == f"{BASE_COLLECTION}__multilingual-e5-large"
    assert collection_name_for(EmbeddingService(model_name=E5, provider="mock")) == f"{BASE_COLLECTION}__multilingual-e5-large"


def test_ensure_collection_enables_idf_on_sparse_vectors():
    client = QdrantClient(":memory:")
    assert ensure_collection(client, "c", vector_size=8) is True
    sparse = client.get_collection("c").config.params.sparse_vectors["sparse"]
    assert sparse.modifier == Modifier.IDF

    ensure_collection(client, "sem_idf", vector_size=8, sparse_idf=False)
    assert client.get_collection("sem_idf").config.params.sparse_vectors["sparse"].modifier in (None, Modifier.NONE)


def test_ensure_collection_is_idempotent_and_rejects_dimension_mismatch():
    client = QdrantClient(":memory:")
    ensure_collection(client, "c", vector_size=8)
    assert ensure_collection(client, "c", vector_size=8) is False
    with pytest.raises(ValueError, match="dimensão 8.*gera 1024"):
        ensure_collection(client, "c", vector_size=1024)
