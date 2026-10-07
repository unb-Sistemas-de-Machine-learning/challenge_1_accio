import logging
import asyncio
import aiohttp
import re
import os
import json
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import unquote, urljoin, urlparse
from typing import Set, List, Optional, TYPE_CHECKING
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from .models import RawDocument, SourceType
from .html import parse_html_content
from .pdf import PDFExtraction, extract_pdf_text

if TYPE_CHECKING:
    from fato_unb.storage.repository import StagingRepository

logger = logging.getLogger(__name__)

DOMAINS = [
    "noticias.unb.br",
    "saa.unb.br",
    "deg.unb.br",
    "dpg.unb.br",
    "adunb.org"
]

START_URLS = [
    "https://adunb.org/categoria/comunicacao/noticias",
    "https://adunb.org/categoria/comunicacao/notas-oficiais",
    "https://noticias.unb.br/ensino",
    "https://noticias.unb.br/informes",
    "https://noticias.unb.br/pesquisas-estudos-e-projetos",
    "https://saa.unb.br/calendario-academico-graduacao/",
    "https://saa.unb.br/calendario-academico-2/",
    "https://deg.unb.br/noticias/",
    "https://dpg.unb.br/category/noticias/"
]

ALLOWED_LISTING_PATHS = [
    '/categoria/comunicacao/noticias',
    '/categoria/comunicacao/notas-oficiais',
    '/ensino',
    '/informes',
    '/pesquisas-estudos-e-projetos',
    '/calendario-academico-graduacao',
    '/calendario-academico-2',
    '/noticias',
    '/category/noticias'
]

ARTICLE_PATTERNS = [
    re.compile(r'^/\d{4}/\d{2}/\d{2}/'),
    re.compile(r'^/\d{4}/\d{2}/'),
    re.compile(r'/\d+-[a-zA-Z0-9-]+'),
    re.compile(r'^/noticias/.+'),
]

CUTOFF_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)

def normalize_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        scheme = "https"
        netloc = parsed.netloc.replace("www.", "")
        path = parsed.path.rstrip('/')
        query = parsed.query
        normalized = f"{scheme}://{netloc}{path}"
        if query:
            normalized += f"?{query}"
        return normalized
    except Exception:
        return url

def load_known_urls(filepath: Optional[str]) -> Set[str]:
    known = set()
    if filepath and os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        url = data.get("url")
                        if url:
                            known.add(normalize_url(url))
                    except Exception:
                        pass
    return known


def load_saved_documents(filepath: Optional[str]) -> List[RawDocument]:
    docs = []
    if filepath and os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        docs.append(RawDocument.model_validate_json(line))
                    except Exception:
                        pass
    return docs


def is_pdf_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.replace("www.", "")
        if netloc not in DOMAINS or parsed.scheme not in ["http", "https"]:
            return False
        path = parsed.path.lower()
        if path.endswith(".pdf"):
            return True
        if any(path.startswith(allowed) for allowed in ALLOWED_LISTING_PATHS):
            return True
        if any(pattern.search(path) for pattern in ARTICLE_PATTERNS):
            return True
        return False
    except Exception:
        return False


def _parse_http_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _pdf_title(extraction: PDFExtraction, url: str) -> str:
    if extraction.title and extraction.title.strip():
        return extraction.title.strip()
    filename = Path(unquote(urlparse(url).path)).stem
    return re.sub(r"[_-]+", " ", filename).strip() or "Documento PDF da UnB"

def check_is_article(url: str) -> bool:
    parsed_path = urlparse(url).path.lower()
    return any(pattern.search(parsed_path) for pattern in ARTICLE_PATTERNS)

def extract_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for a_tag in soup.find_all('a', href=True):
        href = a_tag.get('href')
        if not href:
            continue
        href = href.split('#')[0] 
        full_url = urljoin(base_url, href)
        if is_valid_url(full_url):
            links.append(full_url)
    return links

async def fetch_and_parse(session: aiohttp.ClientSession, url: str) -> tuple:
    try:
        async with session.get(url, timeout=15) as response:
            if response.status == 200:
                content_type = response.headers.get('Content-Type', '')
                if 'application/pdf' in content_type or is_pdf_url(url):
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
                    return (
                        url,
                        extraction.full_text.strip(),
                        [],
                        _pdf_title(extraction, url),
                        published_at,
                    )
                if not content_type.startswith('text/html'):
                    return url, None, [], "", None

                html = await response.text()
                parsed = parse_html_content(html, url)
                links = extract_links(html, url)
                return url, parsed["content"], links, parsed["title"], parsed["published_at"]
    except Exception as e:
        logger.error(f"Erro ao processar {url}: {str(e)}")
    return url, None, [], "", None

async def run_crawler(
    output_file: Optional[str] = "dados.txt",
    repository: Optional["StagingRepository"] = None,
    max_pages: Optional[int] = None,
) -> List[RawDocument]:
    known_urls = load_known_urls(output_file) if output_file else set()
    logger.info(f"Iniciando Crawler. {len(known_urls)} URLs já mapeadas.")
    
    if repository is None:
        from fato_unb.storage.repository import StagingRepository
        repo = StagingRepository()
    else:
        repo = repository

    if output_file and repo is not None:
        saved_file_docs = load_saved_documents(output_file)
        if saved_file_docs:
            await repo.save_documents(saved_file_docs)

    visited: Set[str] = set()
    queue: List[str] = [normalize_url(u) for u in START_URLS]
    in_queue: Set[str] = set(queue)
    saved_count = 0
    all_saved_docs: List[RawDocument] = []
    
    async with aiohttp.ClientSession() as session:
        while queue:
            if max_pages is not None and len(visited) >= max_pages:
                logger.info(f"Limite de {max_pages} páginas visitadas atingido no crawler.")
                break

            batch = queue[:5]
            queue = queue[5:]
            
            logger.info(f"Progresso: {len(visited)} visitadas | {len(queue)} na fila | {saved_count} novas salvas")
            
            tasks = []
            for url in batch:
                if url not in visited:
                    visited.add(url)
                    if (check_is_article(url) or is_pdf_url(url)) and url in known_urls:
                        continue
                    tasks.append(fetch_and_parse(session, url))
            
            if tasks:
                results = await asyncio.gather(*tasks)
                
                batch_docs: List[RawDocument] = []
                for url, text, links, title, published_at in results:
                    is_pdf = is_pdf_url(url)
                    is_candidate = is_pdf or check_is_article(url)
                    min_words = 1 if is_pdf else 50

                    if text and len(text.split()) > min_words and is_candidate:
                        if url not in known_urls:
                            if published_at and published_at >= CUTOFF_DATE:
                                doc = RawDocument(
                                    title=title,
                                    content=text,
                                    url=url,
                                    source=urlparse(url).netloc,
                                    source_type=SourceType.PDF_DOCUMENT if is_pdf else SourceType.HTML_PAGE,
                                    published_at=published_at
                                )
                                batch_docs.append(doc)
                                known_urls.add(url)
                                saved_count += 1
                                logger.debug(f"Salvo: {url} | Data: {published_at}")
                            else:
                                logger.debug(f"Descartado (antigo): {url}")
                    
                    for link in links:
                        norm_link = normalize_url(link)
                        if norm_link not in visited and norm_link not in in_queue:
                            queue.append(norm_link)
                            in_queue.add(norm_link)
                
                if batch_docs:
                    all_saved_docs.extend(batch_docs)
                    if output_file:
                        with open(output_file, "a", encoding="utf-8") as f_out:
                            for doc in batch_docs:
                                f_out.write(doc.model_dump_json() + "\n")
                    if repo is not None:
                        await repo.save_documents(batch_docs)
                
                await asyncio.sleep(3)
                            
    logger.info(f"Execução finalizada. Total de documentos coletados: {len(all_saved_docs)}")
    return all_saved_docs