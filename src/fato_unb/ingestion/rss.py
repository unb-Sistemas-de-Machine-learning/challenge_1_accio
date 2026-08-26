import logging
import feedparser
from datetime import datetime, timezone
from time import mktime
from typing import List
from .models import RawDocument, SourceType

logger = logging.getLogger(__name__)

def fetch_unb_rss_feed(feed_url: str = "https://noticias.unb.br/?format=feed&type=rss") -> List[RawDocument]:
    logger.info(f"Iniciando extração do RSS: {feed_url}")
    parsed_feed = feedparser.parse(feed_url)
    documents = []
    
    for entry in parsed_feed.entries:
        published_parsed = entry.get('published_parsed')
        
        if published_parsed:
            dt_utc = datetime.fromtimestamp(mktime(published_parsed), tz=timezone.utc)
        else:
            dt_utc = datetime.now(timezone.utc)
            
        doc = RawDocument(
            title=entry.get('title', '').strip(),
            content=entry.get('summary', '') or entry.get('description', ''),
            url=entry.get('link', ''),
            source="UnB Notícias",
            source_type=SourceType.RSS_NEWS,
            published_at=dt_utc,
            semester_ref=None
        )
        documents.append(doc)
        
    logger.info(f"Extração RSS concluída. Total de documentos: {len(documents)}")
    return documents