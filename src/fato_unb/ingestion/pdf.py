from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
import re
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
    title: str | None = None
    created_at: datetime | None = None

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


def _parse_pdf_date(value: str | None) -> datetime | None:
    if not value:
        return None

    match = re.fullmatch(
        r"D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?"
        r"(Z|([+-])(\d{2})'?((?:\d{2}))'?)?",
        value,
    )
    if not match:
        return None

    year, month, day, hour, minute, second = match.groups()[:6]
    zone, sign, offset_hour, offset_minute = match.groups()[6:]
    tz = UTC
    if zone and zone != "Z" and sign and offset_hour:
        offset = timedelta(
            hours=int(offset_hour), minutes=int(offset_minute or 0)
        )
        if sign == "-":
            offset = -offset
        tz = timezone(offset)

    try:
        return datetime(
            int(year),
            int(month or 1),
            int(day or 1),
            int(hour or 0),
            int(minute or 0),
            int(second or 0),
            tzinfo=tz,
        ).astimezone(UTC)
    except ValueError:
        return None


def extract_pdf_text(source: str | bytes | bytearray | memoryview) -> PDFExtraction:
    pdf_bytes, source_label = _read_pdf_bytes(source)
    if not pdf_bytes:
        raise ValueError("O conteúdo do PDF está vazio.")

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        pages = [
            PDFPage(page_number=index + 1, text=page.get_text("text", sort=True).strip())
            for index, page in enumerate(document)
        ]
        metadata = document.metadata or {}

    return PDFExtraction(
        source=source_label,
        page_count=len(pages),
        pages=pages,
        title=metadata.get("title") or None,
        created_at=_parse_pdf_date(
            metadata.get("creationDate") or metadata.get("modDate")
        ),
    )
