import pytest
import hashlib
from datetime import datetime, timezone
from pydantic import ValidationError
from src.fato_unb.ingestion.models import RawDocument, SourceType
from src.fato_unb.ingestion.html import fetch_unb_html_document
from src.fato_unb.ingestion.rss import fetch_unb_rss_feed

def test_raw_document_idempotent_hash():
    test_url = "https://noticias.unb.br/exemplo-teste"
    expected_hash = hashlib.sha256(test_url.encode('utf-8')).hexdigest()
    
    doc = RawDocument(
        title="Notícia de Teste",
        content="Conteúdo da notícia de teste.",
        url=test_url,
        source="UnB Notícias",
        source_type=SourceType.RSS_NEWS,
        published_at=datetime.now(timezone.utc)
    )
    
    assert doc.doc_id == expected_hash

def test_raw_document_missing_required_fields():
    with pytest.raises(ValidationError):
        RawDocument(
            title="Notícia Incompleta",
            url="https://noticias.unb.br/incompleta"
        )

def test_extract_html_unb_real_link():
    url = "https://noticias.unb.br/"
    doc = fetch_unb_html_document(url)
    
    assert doc.content is not None
    assert len(doc.content) > 100
    assert doc.url == url
    assert doc.source_type == SourceType.HTML_PAGE
    assert doc.title != "Sem título"

def test_rss_feed_extraction():
    docs = fetch_unb_rss_feed()
    if docs:
        assert isinstance(docs[0], RawDocument)
        assert docs[0].source_type == SourceType.RSS_NEWS
        assert docs[0].title != ""