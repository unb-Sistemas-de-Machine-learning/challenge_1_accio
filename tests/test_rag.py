import pytest

from fato_unb.rag.chunker import SemanticChunker
from fato_unb.rag.embeddings import LEGACY_DENSE_MODEL, EmbeddingService

# Testes do SemanticChunker (TASK-3.1)


def test_chunker_noticia_pequena(mock_noticia_ru):
    """Garante que documentos curtos gerem apenas 1 chunk com metadados corretos."""
    chunker = SemanticChunker(chunk_size=300, overlap_sentences=1)
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
    chunker = SemanticChunker(chunk_size=60, overlap_sentences=1)
    chunks = chunker.chunk_document(mock_edital_extenso)

    assert len(chunks) > 1
    total = len(chunks)

    for idx, chunk in enumerate(chunks):
        assert chunk.chunk_index == idx
        assert chunk.total_chunks == total
        assert chunk.doc_id == mock_edital_extenso.doc_id
        assert len(chunk.chunk_id) == 16
        assert f"[Documento: {mock_edital_extenso.title}]" in chunk.content


def test_chunker_overlap_repete_ultima_frase(mock_edital_extenso):
    """A última frase de um chunk reaparece no início do seguinte (overlap em frases)."""
    chunker = SemanticChunker(chunk_size=40, overlap_sentences=1)
    chunks = chunker.chunk_document(mock_edital_extenso)

    assert len(chunks) >= 2
    ultima_frase = chunks[0].raw_text.rsplit(". ", 1)[-1]
    assert ultima_frase in chunks[1].raw_text


def test_chunker_sem_overlap_nao_repete_frases(mock_edital_extenso):
    chunker = SemanticChunker(chunk_size=40, overlap_sentences=0)
    chunks = chunker.chunk_document(mock_edital_extenso)

    assert len(chunks) >= 2
    ultima_frase = chunks[0].raw_text.rsplit(". ", 1)[-1]
    assert not chunks[1].raw_text.startswith(ultima_frase)


def _doc_com_paragrafos(n_paragrafos: int, frases_por_paragrafo: int = 4):
    from datetime import UTC, datetime

    from fato_unb.ingestion.models import RawDocument, SourceType

    paragrafos = []
    for p in range(n_paragrafos):
        frases = [f"Paragrafo {p} frase {f} traz informacao numero {p * 10 + f} sobre a universidade." for f in range(frases_por_paragrafo)]
        paragrafos.append(" ".join(frases))
    return RawDocument(
        title="UnB Notícias - Titulo de teste",
        content="\n".join(paragrafos),  # trafilatura separa parágrafos com \n simples
        url="https://noticias.unb.br/x/1-teste",
        source="noticias.unb.br",
        source_type=SourceType.HTML_PAGE,
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_chunker_divide_texto_com_quebra_de_linha_simples():
    """Regressão: o corpus real só tem '\\n' (nunca '\\n\\n'), e isso gerava chunks gigantes."""
    doc = _doc_com_paragrafos(n_paragrafos=10)
    chunks = SemanticChunker(chunk_size=60, overlap_sentences=1).chunk_document(doc)

    assert len(chunks) > 3
    assert all(len(c.raw_text.split()) <= 90 for c in chunks)  # tolerância da fusão da cauda
    assert any("\n" in c.raw_text for c in chunks)  # preserva a estrutura de parágrafos


def test_chunker_nao_quebra_frase_em_abreviacoes():
    from fato_unb.rag.chunker import _split_sentences

    frases = _split_sentences("A Profa. Danusa Marques foi indicada. O Art. 2º prevê prazo. Fim.")
    assert frases == ["A Profa. Danusa Marques foi indicada.", "O Art. 2º prevê prazo.", "Fim."]


def test_chunker_parent_text_contem_o_chunk_e_e_maior():
    doc = _doc_com_paragrafos(n_paragrafos=12)
    chunks = SemanticChunker(chunk_size=50, overlap_sentences=1, parent_size=150).chunk_document(doc)

    meio = chunks[len(chunks) // 2]
    assert meio.raw_text in meio.parent_text
    assert len(meio.parent_text.split()) > len(meio.raw_text.split())
    assert len(meio.parent_text.split()) <= 150 + 30  # ~parent_size, com folga de uma frase


def test_chunker_documento_curto_tem_parent_igual_ao_texto(mock_noticia_ru):
    (chunk,) = SemanticChunker().chunk_document(mock_noticia_ru)
    assert chunk.parent_text == mock_noticia_ru.content


def test_chunker_funde_cauda_minuscula_no_chunk_anterior():
    """Um último chunk com poucas palavras novas não deve existir isolado."""
    doc = _doc_com_paragrafos(n_paragrafos=1, frases_por_paragrafo=9)  # 9 frases de 11 palavras
    chunks = SemanticChunker(chunk_size=50, overlap_sentences=1, min_chunk_words=30).chunk_document(doc)

    palavras_novas_ultimo = len(chunks[-1].raw_text.split())
    assert len(chunks) == 1 or palavras_novas_ultimo >= 30


def test_chunker_corta_frase_gigante_por_palavras():
    from datetime import UTC, datetime

    from fato_unb.ingestion.models import RawDocument, SourceType

    doc = RawDocument(
        title="t",
        content=" ".join(["palavra"] * 500),  # uma "frase" de 500 palavras, sem pontuação
        url="https://x.br/a",
        source="x.br",
        source_type=SourceType.HTML_PAGE,
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    chunks = SemanticChunker(chunk_size=100).chunk_document(doc)
    assert len(chunks) >= 5
    assert all(len(c.raw_text.split()) <= 150 for c in chunks)


def test_chunker_limpa_titulo_no_cabecalho_sem_alterar_metadado():
    doc = _doc_com_paragrafos(n_paragrafos=1)
    (chunk, *_) = SemanticChunker().chunk_document(doc)
    assert "[Documento: Titulo de teste]" in chunk.content
    assert chunk.title == "UnB Notícias - Titulo de teste"

    doc.title, doc.source = "Censo da Pós-Graduação – dpg", "dpg.unb.br"
    (chunk, *_) = SemanticChunker().chunk_document(doc)
    assert "[Documento: Censo da Pós-Graduação]" in chunk.content


def test_chunker_ids_sao_deterministicos():
    doc = _doc_com_paragrafos(n_paragrafos=6)
    a = SemanticChunker(chunk_size=50).chunk_document(doc)
    b = SemanticChunker(chunk_size=50).chunk_document(doc)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)


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
    embedder = EmbeddingService(model_name=LEGACY_DENSE_MODEL, provider="local")
    texts = ["Circular normativa do Decanato de Graduação da UnB."]

    vectors = embedder.embed_texts(texts)
    assert len(vectors) == 1
    assert len(vectors[0]) == 384
    assert embedder.vector_dimension == 384

    query_vec = embedder.embed_query("Qual o prazo de matrícula?")
    assert len(query_vec) == 384


def test_sparse_embedding_service_mock():
    """Valida geração de vetores esparsos em modo mock."""
    embedder = EmbeddingService(provider="mock")
    sparse_texts = embedder.embed_sparse_texts(["Texto 1", "Texto 2"])
    assert len(sparse_texts) == 2
    assert "indices" in sparse_texts[0] and "values" in sparse_texts[0]
    assert len(sparse_texts[0]["indices"]) == len(sparse_texts[0]["values"])

    sparse_query = embedder.embed_sparse_query("Consulta teste")
    assert "indices" in sparse_query and "values" in sparse_query
    assert embedder.embed_sparse_texts([]) == []


def test_sparse_embedding_service_local():
    """Valida geração de vetores esparsos (BM25) reais via fastembed."""
    embedder = EmbeddingService(model_name=LEGACY_DENSE_MODEL, provider="local")
    sparse_texts = embedder.embed_sparse_texts(["Restaurante Universitário da UnB"])
    assert len(sparse_texts) == 1
    assert len(sparse_texts[0]["indices"]) > 0
    assert len(sparse_texts[0]["indices"]) == len(sparse_texts[0]["values"])

    sparse_query = embedder.embed_sparse_query("cardápio do RU")
    assert len(sparse_query["indices"]) > 0
    assert len(sparse_query["indices"]) == len(sparse_query["values"])


def test_expand_acronyms():
    """Valida que a expansão de siglas institucionais da UnB funciona com regex de palavras inteiras."""
    from fato_unb.vectorstore.operations import expand_acronyms

    assert "restaurante universitário" in expand_acronyms("Almoço no RU hoje").lower()
    assert "programa de avaliação seriada" in expand_acronyms("Edital do PAS 2026").lower()
    assert "sistema integrado" in expand_acronyms("Acesse o SIGAA").lower()

    # Garante que substrings em palavras não são incorretamente expandidas (ex: RUA)
    assert expand_acronyms("Rua das Oliveiras") == "Rua das Oliveiras"

