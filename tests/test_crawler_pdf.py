import asyncio
import json
from datetime import UTC, datetime

import pymupdf

from fato_unb.ingestion import crawler


class FakeResponse:
    def __init__(self, *, body: bytes = b"", html: str = "", headers=None):
        self.status = 200
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self.body = body
        self.html = html

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    async def read(self) -> bytes:
        return self.body

    async def text(self) -> str:
        return self.html


class FakeSession:
    def __init__(self):
        self.listing_url = "https://saa.unb.br/calendario-academico-graduacao"
        self.pdf_url = "https://saa.unb.br/wp-content/uploads/2026/06/calendario.pdf"
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), "Calendário de graduação do segundo semestre de 2026")
        self.pdf_bytes = document.tobytes()
        document.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    def get(self, url: str, timeout: int):
        if url == self.listing_url:
            return FakeResponse(
                html=(
                    '<html><head><title>Calendário acadêmico</title></head>'
                    '<body><a href="/wp-content/uploads/2026/06/calendario.pdf">'
                    "Calendário 2026.2</a></body></html>"
                )
            )
        if url == self.pdf_url:
            return FakeResponse(
                body=self.pdf_bytes,
                headers={
                    "Content-Type": "application/pdf",
                    "Last-Modified": "Mon, 15 Jun 2026 20:32:00 GMT",
                },
            )
        raise AssertionError(f"URL inesperada no teste: {url}")


def test_crawler_discovers_and_saves_pdf_from_saa(monkeypatch, tmp_path):
    session = FakeSession()
    monkeypatch.setattr(crawler.aiohttp, "ClientSession", lambda: session)
    monkeypatch.setattr(crawler, "START_URLS", [session.listing_url])

    async def no_wait(_seconds: int) -> None:
        return None

    monkeypatch.setattr(crawler.asyncio, "sleep", no_wait)
    output_file = tmp_path / "documents.jsonl"

    asyncio.run(crawler.run_crawler(str(output_file)))

    documents = [
        json.loads(line)
        for line in output_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(documents) == 1
    saved = documents[0]
    assert saved["url"] == session.pdf_url
    assert saved["title"] == "calendario"
    assert saved["source_type"] == "pdf_document"
    assert saved["published_at"].startswith("2026-06-15T20:32:00")
    assert "[Página 1]" in saved["content"]
    assert "Calendário de graduação" in saved["content"]


def test_saa_pdf_links_are_allowed_by_crawler():
    pdf_url = "https://saa.unb.br/wp-content/uploads/2026/06/calendario.pdf"

    assert crawler.is_valid_url(pdf_url)
    assert crawler.is_pdf_url(pdf_url)


def test_pdf_embedded_creation_date_is_available_to_crawler():
    document = pymupdf.open()
    document.new_page()
    document.set_metadata(
        {"title": "Calendário 2026", "creationDate": "D:20260510120000Z"}
    )
    pdf_bytes = document.tobytes()
    document.close()

    extraction = crawler.extract_pdf_text(pdf_bytes)

    assert extraction.title == "Calendário 2026"
    assert extraction.created_at == datetime(2026, 5, 10, 12, 0, tzinfo=UTC)
