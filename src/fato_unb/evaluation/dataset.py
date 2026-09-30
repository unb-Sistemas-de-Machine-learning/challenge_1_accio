import hashlib
import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from fato_unb.ingestion.models import RawDocument
from fato_unb.rag.models import VereditoType

DEFAULT_DATASET = Path(__file__).parent / "retrieval_dataset.jsonl"
DEFAULT_CORPUS = Path(__file__).resolve().parents[3] / "dados.txt"


class TipoCaso(str, Enum):
    VERDADEIRA = "verdadeira"
    FALSA = "falsa"
    DESATUALIZADA = "desatualizada"
    SEM_REGISTRO = "sem_registro"
    PERGUNTA = "pergunta"


class CasoTeste(BaseModel):
    id: str
    tipo: TipoCaso
    alegacao: str = Field(..., description="Afirmação ou pergunta como o usuário escreveria")
    expected_urls: list[str] = Field(
        default_factory=list,
        description="URLs dos documentos que contêm a evidência (vazio em sem_registro)",
    )
    veredito_esperado: VereditoType | None = None
    categoria: str
    dificuldade: str = "media"
    desafio: str = Field(
        "direto",
        description="O que torna o caso difícil: direto, parafrase, coloquial, sigla, "
        "confundidor, numerico, entidade, multi_doc ou temporal",
    )
    evidencia: str = Field("", description="Trecho da fonte que sustenta o rótulo")


def normalize_url(url: str) -> str:
    """Remove fragmento e barra final: '.../x/#main' e '.../x/' são a mesma página."""
    return url.split("#", 1)[0].rstrip("/")


def content_key(doc: RawDocument) -> str:
    return hashlib.md5(doc.content.strip().encode("utf-8")).hexdigest()


def load_dataset(path: Path | str = DEFAULT_DATASET) -> list[CasoTeste]:
    casos = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                casos.append(CasoTeste.model_validate_json(line))
    ids = [c.id for c in casos]
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset contém ids duplicados")
    return casos


def load_corpus(path: Path | str = DEFAULT_CORPUS) -> list[RawDocument]:
    with open(path, encoding="utf-8") as f:
        return [RawDocument.model_validate(json.loads(line)) for line in f if line.strip()]


def dedupe_corpus(docs: list[RawDocument]) -> list[RawDocument]:
    """Mantém um documento por conteúdo idêntico (a primeira ocorrência)."""
    vistos: set[str] = set()
    unicos = []
    for doc in docs:
        key = content_key(doc)
        if key not in vistos:
            vistos.add(key)
            unicos.append(doc)
    return unicos


def url_to_content_keys(docs: list[RawDocument]) -> dict[str, str]:
    return {normalize_url(str(d.url)): content_key(d) for d in docs}


def expected_content_keys(caso: CasoTeste, url_map: dict[str, str]) -> set[str]:
    """Resolve as URLs esperadas para chaves de conteúdo.

    Usar conteúdo (e não doc_id) faz o rótulo valer para cópias do mesmo texto
    com URLs diferentes, e independe de como o corpus foi deduplicado.
    """
    faltando = [u for u in caso.expected_urls if normalize_url(u) not in url_map]
    if faltando:
        raise KeyError(f"Caso {caso.id}: URLs esperadas ausentes do corpus: {faltando}")
    return {url_map[normalize_url(u)] for u in caso.expected_urls}
