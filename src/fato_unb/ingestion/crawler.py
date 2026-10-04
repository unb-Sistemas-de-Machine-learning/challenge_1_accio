import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

from .html import parse_html_content
from .models import RawDocument, SourceType
from .pdf import PDFExtraction, extract_pdf_text

logger = logging.getLogger(__name__)

DOMAINS = ("noticias.unb.br", "saa.unb.br", "deg.unb.br", "dpg.unb.br", "adunb.org")

START_URLS = [
    "https://adunb.org/categoria/comunicacao/noticias",
    "https://adunb.org/categoria/comunicacao/notas-oficiais",
    "https://noticias.unb.br/ensino",
    "https://noticias.unb.br/informes",
    "https://noticias.unb.br/pesquisas-estudos-e-projetos",
    "https://saa.unb.br/calendario-academico-graduacao/",
    "https://saa.unb.br/calendario-academico-2/",
    "https://deg.unb.br/noticias/",
    "https://dpg.unb.br/category/noticias/",
]

ALLOWED_LISTING_PATHS = [
    "/categoria/comunicacao/noticias",
    "/categoria/comunicacao/notas-oficiais",
    "/ensino",
    "/informes",
    "/pesquisas-estudos-e-projetos",
    "/calendario-academico-graduacao",
    "/calendario-academico-2",
    "/noticias",
    "/category/noticias",
]

ARTICLE_PATTERNS = [
    re.compile(r"^/\d{4}/\d{2}/\d{2}/"),
    re.compile(r"^/\d{4}/\d{2}/"),
    re.compile(r"/\d+-[a-zA-Z0-9-]+"),
    re.compile(r"^/noticias/.+"),
]

CUTOFF_DATE = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass(slots=True)
class FetchResult:
    url: str
    content: str | None
    links: list[str]
    title: str
    published_at: datetime | None
    source_type: SourceType | None


def normalize_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        scheme = "https"
        netloc = parsed.netloc.replace("www.", "")
        path = parsed.path.rstrip("/")
        query = parsed.query
        normalized = f"{scheme}://{netloc}{path}"
        if query:
            normalized += f"?{query}"
        return normalized
    except (TypeError, ValueError):
        return url


def load_known_urls(filepath: str) -> set[str]:
    known = set()
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        url = data.get("url")
                        if url:
                            known.add(normalize_url(url))
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        logger.warning(
                            "Ignorando linha inválida em %s: %s", filepath, exc
                        )
    return known


def is_pdf_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.removeprefix("www.")
        if netloc not in DOMAINS or parsed.scheme not in ("http", "https"):
            return False

        path = parsed.path.lower()
        return (
            path.endswith(".pdf")
            or any(path.startswith(allowed) for allowed in ALLOWED_LISTING_PATHS)
            or any(pattern.search(path) for pattern in ARTICLE_PATTERNS)
        )
    except (TypeError, ValueError):
        return False


def check_is_article(url: str) -> bool:
    parsed_path = urlparse(url).path.lower()
    return any(pattern.search(parsed_path) for pattern in ARTICLE_PATTERNS)


def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag.get("href")
        if not href:
            continue
        href = href.split("#")[0]
        full_url = urljoin(base_url, href)
        if is_valid_url(full_url):
            links.append(full_url)
    return links


def _parse_http_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _pdf_title(extraction: PDFExtraction, url: str) -> str:
    if extraction.title and extraction.title.strip():
        return extraction.title.strip()
    filename = Path(unquote(urlparse(url).path)).stem
    return re.sub(r"[_-]+", " ", filename).strip() or "Documento PDF da UnB"


async def fetch_and_parse(session: aiohttp.ClientSession, url: str) -> FetchResult:
    try:
        async with session.get(url, timeout=15) as response:
            if response.status != 200:
                logger.warning("Portal respondeu HTTP %s para %s", response.status, url)
                return FetchResult(url, None, [], "", None, None)

            content_type = response.headers.get("Content-Type", "").lower()
            if "application/pdf" in content_type or is_pdf_url(url):
                pdf_bytes = await response.read()
                extraction = await asyncio.to_thread(extract_pdf_text, pdf_bytes)
                published_at = (
                    _parse_http_date(response.headers.get("Last-Modified"))
                    or extraction.created_at
                )
                if published_at is None:
                    logger.warning(
                        "PDF sem data de publicação nos cabeçalhos ou metadados; ignorando: %s",
                        url,
                    )
                return FetchResult(
                    url=url,
                    content=extraction.full_text.strip(),
                    links=[],
                    title=_pdf_title(extraction, url),
                    published_at=published_at,
                    source_type=SourceType.PDF_DOCUMENT,
                )

            if not content_type.startswith("text/html"):
                return FetchResult(url, None, [], "", None, None)

            html = await response.text()
            parsed = parse_html_content(html, url)
            links = extract_links(html, url)
            return FetchResult(
                url=url,
                content=parsed["content"],
                links=links,
                title=parsed["title"],
                published_at=parsed["published_at"],
                source_type=SourceType.HTML_PAGE,
            )
    except (TimeoutError, aiohttp.ClientError, ValueError):
        logger.exception("Erro ao processar %s", url)
    except Exception:
        logger.exception("Erro inesperado ao processar %s", url)
    return FetchResult(url, None, [], "", None, None)


async def run_crawler(output_file: str = "dados.txt"):
    known_urls = load_known_urls(output_file)
    logger.info("Iniciando Crawler. %s URLs já mapeadas.", len(known_urls))

    visited: set[str] = set()
    queue: list[str] = [normalize_url(u) for u in START_URLS]
    in_queue: set[str] = set(queue)
    saved_count = 0

    async with aiohttp.ClientSession() as session:
        while queue:
            batch = queue[:5]
            queue = queue[5:]

            logger.info(
                "Progresso: %s visitadas | %s na fila | %s novas salvas",
                len(visited),
                len(queue),
                saved_count,
            )

            tasks = []
            for url in batch:
                if url not in visited:
                    visited.add(url)
                    if (check_is_article(url) or is_pdf_url(url)) and url in known_urls:
                        continue
                    tasks.append(fetch_and_parse(session, url))

            if tasks:
                results = await asyncio.gather(*tasks)

                with open(output_file, "a", encoding="utf-8") as f_out:
                    for result in results:
                        is_pdf = result.source_type == SourceType.PDF_DOCUMENT
                        is_candidate = is_pdf or check_is_article(result.url)
                        min_words = 1 if is_pdf else 50

                        if (
                            result.content
                            and len(result.content.split()) >= min_words
                            and is_candidate
                            and result.url not in known_urls
                            and result.published_at
                            and result.published_at >= CUTOFF_DATE
                        ):
                            doc = RawDocument(
                                title=result.title,
                                content=result.content,
                                url=result.url,
                                source=urlparse(result.url).netloc,
                                source_type=result.source_type,
                                published_at=result.published_at,
                            )
                            f_out.write(doc.model_dump_json() + "\n")
                            known_urls.add(result.url)
                            saved_count += 1
                            logger.info(
                                "Salvo %s: %s | Data: %s",
                                result.source_type.value,
                                result.url,
                                result.published_at,
                            )
                        elif (
                            result.content
                            and is_candidate
                            and result.url not in known_urls
                            and result.published_at
                            and result.published_at < CUTOFF_DATE
                        ):
                            logger.debug("Documento antigo descartado: %s", result.url)

                        for link in result.links:
                            norm_link = normalize_url(link)
                            if norm_link not in visited and norm_link not in in_queue:
                                queue.append(norm_link)
                                in_queue.add(norm_link)

                await asyncio.sleep(3)

    logger.info("Execução finalizada. %s novos documentos salvos.", saved_count)
