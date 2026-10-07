import hashlib
from datetime import datetime
from enum import Enum
from typing import Optional, Union
from pydantic import BaseModel, Field, HttpUrl, model_validator

class SourceType(str, Enum):
    RSS_NEWS = "rss_news"
    HTML_PAGE = "html_page"
    PDF_DOCUMENT = "pdf_document"

class RawDocument(BaseModel):
    title: str
    content: str
    url: Union[str, HttpUrl]
    source: str
    source_type: SourceType
    published_at: datetime
    semester_ref: Optional[str] = None
    doc_id: str = Field(default="")

    @model_validator(mode='after')
    def set_derived_fields(self) -> 'RawDocument':
        if not self.doc_id:
            url_str = str(self.url)
            self.doc_id = hashlib.sha256(url_str.encode('utf-8')).hexdigest()
            
        if not self.semester_ref and self.published_at:
            year = self.published_at.year
            semester = 1 if self.published_at.month <= 7 else 2
            self.semester_ref = f"{year}.{semester}"
            
        return self