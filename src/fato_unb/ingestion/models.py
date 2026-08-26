import hashlib
from enum import Enum
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, model_validator

class SourceType(str, Enum):
    RSS_NEWS = "rss_news"
    HTML_PAGE = "html_page"
    PDF_DOCUMENT = "pdf_document"

class RawDocument(BaseModel):
    title: str
    content: str
    url: str
    source: str
    source_type: SourceType
    published_at: datetime
    semester_ref: Optional[str] = None
    doc_id: str = Field(default="")

    @model_validator(mode='after')
    def generate_doc_id(self) -> 'RawDocument':
        if not self.doc_id:
            self.doc_id = hashlib.sha256(self.url.encode('utf-8')).hexdigest()
        return self