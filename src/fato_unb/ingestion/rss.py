import asyncio
import logging
import feedparser
from datetime import datetime, timezone
from time import mktime
from typing import List, Optional, TYPE_CHECKING
from .html import extract_html_data
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

def enrich_documents_with_full_text(docs: List[RawDocument]) -> int:
    """Troca o resumo do RSS pelo texto completo da notícia, alterando os documentos no lugar.

    O feed só traz o resumo (17 a 30 palavras), o que deixa a notícia mais recente como uma
    evidência fraca. Se o download ou a extração falhar, ou o texto vier menor que o resumo,
    o resumo é mantido. Devolve quantos documentos foram enriquecidos.
    """
    enriched = 0
    for doc in docs:
        try:
            full_text = (extract_html_data(str(doc.url)).get("content") or "").strip()
        except Exception as exc:
            logger.warning(f"RSS: texto completo indisponível para {doc.url} ({exc}); mantendo o resumo.")
            continue
        if len(full_text.split()) > len(doc.content.split()):
            doc.content = full_text
            enriched += 1
    logger.info(f"RSS: {enriched}/{len(docs)} documentos enriquecidos com o texto completo.")
    return enriched


async def enrich_new_documents(
    docs: List[RawDocument], repository: "StagingRepository"
) -> int:
    """Enriquece só o que ainda não está no staging (o resto seria baixado toda hora à toa).

    O download é síncrono, então roda numa thread para não travar o loop do agendador."""
    existing = await repository.get_existing_doc_ids([d.doc_id for d in docs])
    novos = [d for d in docs if d.doc_id not in existing]
    if not novos:
        return 0
    return await asyncio.to_thread(enrich_documents_with_full_text, novos)


async def ingest_unb_rss_feed(
    feed_url: str = "https://noticias.unb.br/?format=feed&type=rss",
    repository: Optional["StagingRepository"] = None,
    fetch_full_text: bool = True,
) -> List[RawDocument]:
    docs = fetch_unb_rss_feed(feed_url=feed_url)
    if repository is None:
        from fato_unb.storage.repository import StagingRepository
        repo = StagingRepository()
    else:
        repo = repository
    if docs and fetch_full_text:
        await enrich_new_documents(docs, repo)
    if docs:
        await repo.save_documents(docs)
    return docs