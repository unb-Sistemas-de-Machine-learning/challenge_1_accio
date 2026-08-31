import logging
import re
from datetime import UTC, datetime

import trafilatura
from bs4 import BeautifulSoup

from .models import RawDocument, SourceType

logger = logging.getLogger(__name__)


def parse_html_content(html_content: str, url: str) -> dict:
    soup = BeautifulSoup(html_content, "html.parser")

    for element in soup.find_all(
        attrs={"class": lambda c: c and "cookie" in c.lower()}
    ):
        element.decompose()

    text = trafilatura.extract(
        html_content, include_comments=False, include_tables=False
    )

    if not text:
        for element in soup(
            ["script", "style", "nav", "footer", "header", "aside", "form", "button"]
        ):
            element.decompose()
        text = soup.get_text(separator="\n", strip=True)

    title_tag = soup.title
    title = title_tag.string.strip() if title_tag and title_tag.string else "Sem título"

    published_at = None
    meta_date = soup.find("meta", property="article:published_time")

    if meta_date and meta_date.get("content"):
        date_str = meta_date.get("content")
    else:
        time_tag = soup.find("time")
        date_str = time_tag.get("datetime") if time_tag else None

    if not date_str:
        url_date_match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", url)
        if url_date_match:
            date_str = f"{url_date_match.group(1)}-{url_date_match.group(2)}-{url_date_match.group(3)}"

    if date_str:
        try:
            if "T" in date_str:
                published_at = datetime.fromisoformat(date_str)
            else:
                published_at = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(
                    tzinfo=UTC
                )
        except (TypeError, ValueError):
            logger.warning(
                f"Falha ao analisar a data de publicação: {date_str} para a URL: {url}"
            )

    return {"title": title, "content": text, "url": url, "published_at": published_at}


def extract_html_data(url: str) -> dict:
    logger.info(f"Baixando conteúdo HTML da URL: {url}")
    html_content = trafilatura.fetch_url(url)
    if not html_content:
        raise ValueError("Falha no download")
    return parse_html_content(html_content, url)


def fetch_unb_html_document(
    url: str, source: str = "UnB Notícias"
) -> RawDocument | None:
    data = extract_html_data(url)
    if not data.get("published_at"):
        logger.info(f"Ignorando documento sem data de publicação: {url}")
        return None

    return RawDocument(
        title=data["title"],
        content=data["content"],
        url=data["url"],
        source=source,
        source_type=SourceType.HTML_PAGE,
        published_at=data["published_at"],
    )
