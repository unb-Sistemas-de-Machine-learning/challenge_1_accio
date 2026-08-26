import logging
import asyncio
import aiohttp
import re
import os
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import Set, List
from datetime import datetime, timezone
from .models import RawDocument, SourceType
from .html import parse_html_content

logger = logging.getLogger(__name__)

DOMAINS = [
    "noticias.unb.br",
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
    "https://deg.unb.br/noticias/",
    "https://dpg.unb.br/category/noticias/"
]

ALLOWED_LISTING_PATHS = [
    '/categoria/comunicacao/noticias',
    '/categoria/comunicacao/notas-oficiais',
    '/ensino',
    '/informes',
    '/pesquisas-estudos-e-projetos',
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

def load_known_urls(filepath: str) -> Set[str]:
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
                    except Exception:
                        pass
    return known

def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.replace("www.", "")
        if netloc not in DOMAINS or parsed.scheme not in ["http", "https"]:
            return False
        path = parsed.path.lower()
        if any(path.startswith(allowed) for allowed in ALLOWED_LISTING_PATHS):
            return True
        if any(pattern.search(path) for pattern in ARTICLE_PATTERNS):
            return True
        return False
    except Exception:
        return False

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
                if 'application/pdf' in content_type:
                    return url, "PDF_DOCUMENT", [], "", None
                if not content_type.startswith('text/html'):
                    return url, None, [], "", None
                    
                html = await response.text()
                parsed = parse_html_content(html, url)
                links = extract_links(html, url)
                return url, parsed["content"], links, parsed["title"], parsed["published_at"]
    except Exception as e:
        logger.error(f"Erro ao processar {url}: {str(e)}")
    return url, None, [], "", None

async def run_crawler(output_file: str = "dados.txt"):
    known_urls = load_known_urls(output_file)
    logger.info(f"Iniciando Crawler. {len(known_urls)} URLs já mapeadas.")
    
    visited: Set[str] = set()
    queue: List[str] = [normalize_url(u) for u in START_URLS]
    in_queue: Set[str] = set(queue)
    saved_count = 0
    
    async with aiohttp.ClientSession() as session:
        while queue:
            batch = queue[:5]
            queue = queue[5:]
            
            logger.info(f"Progresso: {len(visited)} visitadas | {len(queue)} na fila | {saved_count} novas salvas")
            
            tasks = []
            for url in batch:
                if url not in visited:
                    visited.add(url)
                    if check_is_article(url) and url in known_urls:
                        continue
                    tasks.append(fetch_and_parse(session, url))
            
            if tasks:
                results = await asyncio.gather(*tasks)
                
                with open(output_file, "a", encoding="utf-8") as f_out:
                    for url, text, links, title, published_at in results:
                        is_article = check_is_article(url)
                        
                        if text and text != "PDF_DOCUMENT" and len(text.split()) > 50 and is_article:
                            if url not in known_urls:
                                if published_at and published_at >= CUTOFF_DATE:
                                    doc = RawDocument(
                                        title=title,
                                        content=text,
                                        url=url,
                                        source=urlparse(url).netloc,
                                        source_type=SourceType.HTML_PAGE,
                                        published_at=published_at
                                    )
                                    f_out.write(doc.model_dump_json() + "\n")
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
                
                await asyncio.sleep(3)
                            
    logger.info("Execução finalizada.")