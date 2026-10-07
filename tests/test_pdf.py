from __future__ import annotations

from dataclasses import dataclass

import pymupdf
import pytest
import requests

from fato_unb.ingestion.pdf import extract_pdf_text


def make_pdf_bytes(*page_texts: str) -> bytes:
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_extract_pdf_text_from_bytes_preserves_pages():
    pdf_bytes = make_pdf_bytes("Calendário acadêmico 2026", "Matrículas em março")

    result = extract_pdf_text(pdf_bytes)

    assert result.source == "bytes"
    assert result.page_count == 2
    assert [page.page_number for page in result.pages] == [1, 2]
    assert "Calendário acadêmico 2026" in result.pages[0].text
    assert "Matrículas em março" in result.pages[1].text
    assert "[Página 1]" in result.full_text
    assert "[Página 2]" in result.full_text


def test_extract_pdf_text_from_url(monkeypatch: pytest.MonkeyPatch):
    pdf_url = "https://noticias.unb.br/documentos/calendario.pdf"
    pdf_bytes = make_pdf_bytes("Calendário acadêmico UnB")

    @dataclass
    class FakeResponse:
        content: bytes
        headers: dict[str, str]

        def raise_for_status(self) -> None:
            return None

    requested: list[tuple[str, int]] = []

    def fake_get(url: str, timeout: int) -> FakeResponse:
        requested.append((url, timeout))
        return FakeResponse(pdf_bytes, {"Content-Type": "application/pdf"})

    monkeypatch.setattr("fato_unb.ingestion.pdf.requests.get", fake_get)

    result = extract_pdf_text(pdf_url)

    assert requested == [(pdf_url, 30)]
    assert result.source == pdf_url
    assert result.page_count == 1
    assert "Calendário acadêmico UnB" in result.pages[0].text


def test_extract_pdf_text_rejects_non_pdf_url(monkeypatch: pytest.MonkeyPatch):
    @dataclass
    class FakeResponse:
        content: bytes = b"<html>not a PDF</html>"
        headers: dict[str, str] | None = None

        def __post_init__(self) -> None:
            if self.headers is None:
                self.headers = {"Content-Type": "text/html"}

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(
        "fato_unb.ingestion.pdf.requests.get",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(ValueError, match="não retornou um arquivo PDF"):
        extract_pdf_text("https://noticias.unb.br/documentos/pagina")


def test_extract_pdf_text_propagates_http_error(monkeypatch: pytest.MonkeyPatch):
    class FakeResponse:
        def raise_for_status(self) -> None:
            raise requests.HTTPError("404 Not Found")

    monkeypatch.setattr(
        "fato_unb.ingestion.pdf.requests.get",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(requests.HTTPError, match="404"):
        extract_pdf_text("https://noticias.unb.br/documentos/inexistente.pdf")


def test_extract_pdf_text_rejects_empty_bytes():
    with pytest.raises(ValueError, match="vazio"):
        extract_pdf_text(b"")
