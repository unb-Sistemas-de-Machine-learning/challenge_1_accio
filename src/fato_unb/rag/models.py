from enum import Enum

from pydantic import BaseModel, Field, HttpUrl


class VereditoType(str, Enum):
    CONFIRMADO_OFICIALMENTE = "CONFIRMADO_OFICIALMENTE"
    BOATO_SEM_REGISTRO = "BOATO_SEM_REGISTRO"
    DESATUALIZADO_OU_FORA_DE_CONTEXTO = "DESATUALIZADO_OU_FORA_DE_CONTEXTO"
    INCONCLUSIVO = "INCONCLUSIVO"


class DocumentChunk(BaseModel):
    chunk_id: str = Field(
        ..., description="Hash determinístico: sha256(doc_id + chunk_index)"
    )
    doc_id: str = Field(..., description="ID do documento original")
    content: str = Field(
        ..., description="Texto enriquecido com metadados para busca vetorial"
    )
    raw_text: str = Field(..., description="Trecho puro sem cabeçalhos")
    chunk_index: int = Field(..., description="Posição do chunk no documento")
    total_chunks: int = Field(..., description="Total de chunks gerados para este doc")

    # Metadados para filtros
    title: str
    url: HttpUrl
    source: str
    semester_ref: str | None = None


class FonteCitada(BaseModel):
    title: str
    url: HttpUrl
    source: str


class VereditoJSON(BaseModel):
    veredito: VereditoType
    justificativa: str
    fontes: list[FonteCitada] = Field(default_factory=list)
    confianca: float = Field(..., ge=0.0, le=1.0)
    afirmacao_analisada: str
