import logging
import feedparser
from datetime import datetime, timezone
from time import mktime
from typing import List, Optional, TYPE_CHECKING
from .models import RawDocument, SourceType
from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from fato_unb.storage.repository import StagingRepository

logger = logging.getLogger(__name__)


def clean_html_text(raw_html: str) -> str:
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def fetch_unb_rss_feed(feed_url: str = "https://noticias.unb.br/?format=feed&type=rss") -> List[RawDocument]:
    logger.info(f"Iniciando extração do RSS: {feed_url}")
    parsed_feed = feedparser.parse(feed_url)
    documents = []

    for entry in parsed_feed.entries:
        published_parsed = entry.get("published_parsed")

        if published_parsed:
            dt_utc = datetime.fromtimestamp(mktime(published_parsed), tz=timezone.utc)
        else:
            dt_utc = datetime.now(timezone.utc)

        raw_content = entry.get("summary", "") or entry.get("description", "")
        clean_content = clean_html_text(raw_content)

        doc = RawDocument(
            title=entry.get("title", "").strip(),
            content=clean_content,
            url=entry.get("link", ""),
            source="UnB Notícias",
            source_type=SourceType.RSS_NEWS,
            published_at=dt_utc,
            semester_ref=None,
        )
        documents.append(doc)
        
    logger.info(f"Extração RSS concluída. Total de documentos: {len(documents)}")
    return documents

async def ingest_unb_rss_feed(
    feed_url: str = "https://noticias.unb.br/?format=feed&type=rss",
    repository: Optional["StagingRepository"] = None,
) -> List[RawDocument]:
    docs = fetch_unb_rss_feed(feed_url=feed_url)
    if repository is None:
        from fato_unb.storage.repository import StagingRepository
        repo = StagingRepository()
    else:
        repo = repository
    if docs:
        await repo.save_documents(docs)
    return docs