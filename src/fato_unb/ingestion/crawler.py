import logging
import asyncio
import aiohttp
import re
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
    "www.deg.unb.br",
    "dpg.unb.br",
    "www.dpg.unb.br",
    "adunb.org",
    "www.adunb.org"
]

START_URLS = [
    "https://adunb.org/categoria/comunicacao/noticias",
    "https://adunb.org/categoria/comunicacao/notas-oficiais",
    "https://noticias.unb.br/ensino",
    "https://noticias.unb.br/informes",
    "https://noticias.unb.br/pesquisas-estudos-e-projetos",
    "https://www.deg.unb.br/noticias/",
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

def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.netloc not in DOMAINS or parsed.scheme not in ["http", "https"]:
            return False
            
        path = parsed.path.lower()
        
        if any(path.startswith(allowed) for allowed in ALLOWED_LISTING_PATHS):
            return True
            
        if any(pattern.search(path) for pattern in ARTICLE_PATTERNS):
            return True
            
        return False
    except Exception:
        return False

def extract_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for a_tag in soup.find_all('a', href=True):
        href = a_tag.get('href')
        if not href:
            continue
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
                    return url, "PDF_DOCUMENT", [], ""
                if not content_type.startswith('text/html'):
                    return url, None, [], ""
                    
                html = await response.text()
                parsed = parse_html_content(html, url)
                links = extract_links(html, url)
                return url, parsed["content"], links, parsed["title"]
            else:
                logger.warning(f"Status {response.status} ao acessar: {url}")
    except Exception as e:
        logger.error(f"Erro ao processar {url}: {str(e)}")
    return url, None, [], ""

async def run_crawler(output_file: str = "dados.txt"):
    logger.info(f"Iniciando Crawler Assíncrono. Destino: {output_file}")
    visited: Set[str] = set()
    queue: List[str] = START_URLS.copy()
    
    async with aiohttp.ClientSession() as session:
        while queue:
            batch = queue[:20]
            queue = queue[20:]
            
            logger.info(f"Processando lote com {len(batch)} URLs. Fila pendente: {len(queue)}")
            
            tasks = []
            for url in batch:
                if url not in visited:
                    visited.add(url)
                    tasks.append(fetch_and_parse(session, url))
            
            results = await asyncio.gather(*tasks)
            
            with open(output_file, "a", encoding="utf-8") as f:
                for url, text, links, title in results:
                    parsed_path = urlparse(url).path.lower()
                    is_article = any(pattern.search(parsed_path) for pattern in ARTICLE_PATTERNS)
                    
                    if text and text != "PDF_DOCUMENT" and len(text.split()) > 50 and is_article:
                        doc = RawDocument(
                            title=title,
                            content=text,
                            url=url,
                            source=urlparse(url).netloc,
                            source_type=SourceType.HTML_PAGE,
                            published_at=datetime.now(timezone.utc)
                        )
                        f.write(doc.model_dump_json() + "\n")
                        logger.debug(f"Documento extraído com sucesso: {url}")
                    
                    for link in links:
                        if link not in visited and link not in queue:
                            queue.append(link)
                            
    logger.info("Execução do Crawler finalizada.")