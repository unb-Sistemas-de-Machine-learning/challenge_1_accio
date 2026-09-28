from __future__ import annotations

from typing import Literal

import pymupdf
import requests
from pydantic import BaseModel, Field


class PDFPage(BaseModel):

    page_number: int = Field(ge=1)
    text: str


class PDFExtraction(BaseModel):

    source: str | Literal["bytes"]
    page_count: int = Field(ge=0)
    pages: list[PDFPage]

    @property
    def full_text(self) -> str:
        return "\n\n".join(
            f"[Página {page.page_number}]\n{page.text}" for page in self.pages
        )


def _read_pdf_bytes(source: str | bytes | bytearray | memoryview) -> tuple[bytes, str]:
    if isinstance(source, str):
        if not source.startswith(("http://", "https://")):
            raise ValueError("A origem textual deve ser uma URL HTTP ou HTTPS.")
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" not in content_type and not response.content.startswith(
            b"%PDF-"
        ):
            raise ValueError(f"A URL não retornou um arquivo PDF: {source}")
        return response.content, source

    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source), "bytes"

    raise TypeError("source deve ser uma URL HTTP(S) ou bytes de um PDF.")


def extract_pdf_text(source: str | bytes | bytearray | memoryview) -> PDFExtraction:
    pdf_bytes, source_label = _read_pdf_bytes(source)
    if not pdf_bytes:
        raise ValueError("O conteúdo do PDF está vazio.")

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        pages = [
            PDFPage(page_number=index + 1, text=page.get_text("text", sort=True).strip())
            for index, page in enumerate(document)
        ]

    return PDFExtraction(source=source_label, page_count=len(pages), pages=pages)
