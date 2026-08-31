import logging
from datetime import UTC, datetime
from time import mktime

import feedparser

from .models import RawDocument, SourceType

logger = logging.getLogger(__name__)


def fetch_unb_rss_feed(
    feed_url: str = "https://noticias.unb.br/?format=feed&type=rss",
) -> list[RawDocument]:
    logger.info(f"Iniciando extração do RSS: {feed_url}")
    parsed_feed = feedparser.parse(feed_url)
    documents = []

    for entry in parsed_feed.entries:
        published_parsed = entry.get("published_parsed")

        if not published_parsed:
            logger.warning(
                f"Entrada RSS sem data de publicação: {entry.get('title', 'Sem título')}"
            )
            continue
        dt_utc = datetime.fromtimestamp(mktime(published_parsed), tz=UTC)

        doc = RawDocument(
            title=entry.get("title", "").strip(),
            content=entry.get("summary", "") or entry.get("description", ""),
            url=entry.get("link", ""),
            source="UnB Notícias",
            source_type=SourceType.RSS_NEWS,
            published_at=dt_utc,
            semester_ref=None,
        )
        documents.append(doc)

    logger.info(f"Extração RSS concluída. Total de documentos: {len(documents)}")
    return documents
