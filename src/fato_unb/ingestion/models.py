import hashlib
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl, computed_field


class SourceType(str, Enum):
    RSS_NEWS = "rss_news"
    HTML_PAGE = "html_page"
    PDF_DOCUMENT = "pdf_document"


class RawDocument(BaseModel):
    title: str = Field(..., description="Título da matéria, edital ou comunicado")
    content: str = Field(..., description="Texto limpo extraído da fonte")
    url: HttpUrl = Field(..., description="URL canônica de origem")
    source: str = Field(..., description="Órgão emissor (ex: UnB Notícias, DEG, SAA)")
    source_type: SourceType
    published_at: datetime | None = Field(default=None)
    semester_ref: str | None = Field(default=None)

    @computed_field
    @property
    def doc_id(self) -> str:
        return hashlib.sha256(str(self.url).encode("utf-8")).hexdigest()[:16]
