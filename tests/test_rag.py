import pytest

from fato_unb.rag.chunker import SemanticChunker
from fato_unb.rag.embeddings import EmbeddingService

# Testes do SemanticChunker (TASK-3.1)


def test_chunker_noticia_pequena(mock_noticia_ru):
    """Garante que documentos curtos gerem apenas 1 chunk com metadados corretos."""
    chunker = SemanticChunker(chunk_size=300, chunk_overlap=30)
    chunks = chunker.chunk_document(mock_noticia_ru)

    assert len(chunks) == 1
    chunk = chunks[0]

    assert chunk.doc_id == mock_noticia_ru.doc_id
    assert chunk.chunk_index == 0
    assert chunk.total_chunks == 1
    assert chunk.title == mock_noticia_ru.title
    assert chunk.source == "UnB Notícias"
    assert chunk.semester_ref == "2026/1"

    # Valida injeção do cabeçalho contextual no content
    assert "[Documento: Funcionamento do RU no Feriado]" in chunk.content
    assert "[Fonte: UnB Notícias | Ref: 2026/1]" in chunk.content
    assert chunk.raw_text == mock_noticia_ru.content


def test_chunker_documento_extenso_preserva_ordem_e_totais(mock_edital_extenso):
    """Garante divisão em múltiplos blocos respeitando sequência e totais."""
    chunk_size = 60
    chunk_overlap = 15
    chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_document(mock_edital_extenso)

    assert len(chunks) > 1
    total = len(chunks)

    for idx, chunk in enumerate(chunks):
        assert chunk.chunk_index == idx
        assert chunk.total_chunks == total
        assert chunk.doc_id == mock_edital_extenso.doc_id
        assert len(chunk.chunk_id) == 16
        assert f"[Documento: {mock_edital_extenso.title}]" in chunk.content


def test_chunker_overlap_mantem_continuidade(mock_edital_extenso):
    """Valida se as últimas palavras do chunk anterior aparecem no próximo chunk."""
    chunker = SemanticChunker(chunk_size=40, chunk_overlap=10)
    chunks = chunker.chunk_document(mock_edital_extenso)

    assert len(chunks) >= 2

    # Pega palavras do final do primeiro chunk e checa presença no segundo
    palavras_chunk_0 = chunks[0].raw_text.split()
    fim_chunk_0 = " ".join(palavras_chunk_0[-3:])

    assert fim_chunk_0 in chunks[1].raw_text


def test_chunker_documento_vazio(mock_noticia_ru):
    """Valida comportamento seguro ao receber documento com conteúdo vazio."""
    mock_noticia_ru.content = ""
    chunker = SemanticChunker()
    chunks = chunker.chunk_document(mock_noticia_ru)

    assert len(chunks) == 0


# Testes do EmbeddingService


def test_mock_embedding_service():
    """Valida execução instantânea sem modelo carregado."""
    embedder = EmbeddingService(provider="mock", mock_dimension=384)
    texts = ["Texto de teste 1", "Texto de teste 2"]

    vectors = embedder.embed_texts(texts)
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert vectors[0][0] == 0.05
    assert embedder.vector_dimension == 384

    query_vec = embedder.embed_query("Consulta teste")
    assert len(query_vec) == 384
    assert isinstance(query_vec[0], float)


def test_embedding_service_lista_vazia():
    """Valida retorno de lista vazia sem erro."""
    embedder = EmbeddingService(provider="mock")
    assert embedder.embed_texts([]) == []


def test_embedding_service_provedor_invalido():
    """Valida levantamento de ValueError para provider não mapeado."""
    with pytest.raises(ValueError, match="não suportado"):
        EmbeddingService(provider="invalido")


def test_local_fastembed_service():
    """Testa geração real de vetores via fastembed (ONNX)."""
    embedder = EmbeddingService(provider="local")
    texts = ["Circular normativa do Decanato de Graduação da UnB."]

    vectors = embedder.embed_texts(texts)
    assert len(vectors) == 1
    assert len(vectors[0]) == 384
    assert embedder.vector_dimension == 384

    query_vec = embedder.embed_query("Qual o prazo de matrícula?")
    assert len(query_vec) == 384
