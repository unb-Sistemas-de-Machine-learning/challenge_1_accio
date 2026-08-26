import logging
import trafilatura
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from .models import RawDocument, SourceType

logger = logging.getLogger(__name__)

def parse_html_content(html_content: str, url: str) -> dict:
    text = trafilatura.extract(html_content)
    soup = BeautifulSoup(html_content, 'html.parser')
    
    if not text:
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        text = soup.get_text(separator='\n', strip=True)
        
    title_tag = soup.title
    title = title_tag.string.strip() if title_tag and title_tag.string else "Sem título"
    
    return {
        "title": title,
        "content": text,
        "url": url
    }

def extract_html_data(url: str) -> dict:
    logger.info(f"Baixando conteúdo HTML da URL: {url}")
    html_content = trafilatura.fetch_url(url)
    if not html_content:
        logger.error(f"Falha no download da URL: {url}")
        raise ValueError("Falha no download")
    return parse_html_content(html_content, url)

def fetch_unb_html_document(url: str, source: str = "UnB Notícias") -> RawDocument:
    data = extract_html_data(url)
    
    return RawDocument(
        title=data["title"],
        content=data["content"],
        url=data["url"],
        source=source,
        source_type=SourceType.HTML_PAGE,
        published_at=datetime.now(timezone.utc)
    )