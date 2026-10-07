# Pipeline Unificado de Ingestão e Indexação Vetorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrar os módulos de Ingestão, Staging Relacional (PostgreSQL via SQLAlchemy 2.0 Async), Fatiamento Semântico (SemanticChunker), Embeddings Locais (FastEmbed) e Banco Vetorial (Qdrant) em um pipeline desacoplado, idempotente e resiliente.

**Architecture:** Arquitetura desacoplada em duas etapas: Ingestão salva documentos oficiais brutos (`RawDocument`) no PostgreSQL/SQLite com status `PENDING` e controle de deduplicação por hash (`doc_id`). Em seguida, o `IndexingPipeline` recupera os pendentes, quebra em `DocumentChunk` com injeção de contexto de procedência, calcula embeddings densos 384d em lotes via FastEmbed e persiste no Qdrant com UUIDs determinísticos, atualizando o status do documento para `INDEXED` ou `FAILED`.

**Tech Stack:** Python 3.14+, SQLAlchemy 2.0 Async, aiosqlite, asyncpg, Qdrant Client 1.19+, FastEmbed (ONNX Runtime), Pydantic v2, pytest.

**Spec:** [`docs/superpowers/specs/2026-09-16-unified-indexing-pipeline-design.md`](../specs/2026-09-16-unified-indexing-pipeline-design.md)

## Global Constraints
- Python >= 3.14 obrigatório.
- Gerenciamento estrito de dependências via `uv`.
- Contratos estritos com Pydantic v2 para todas as transições de dados (`RawDocument`, `DocumentChunk`).
- Injeção obrigatória de contexto no conteúdo de cada chunk (`[Documento: ...] [Fonte: ...] [Ref: ...]`).
- Idempotência estrita: re-execuções não devem duplicar registros no banco relacional nem no Qdrant (UUIDs RFC-4122 determinísticos baseados no hash do chunk).
- Isolamento de testes unitários: testes rápidos sem dependência de containers ou rede externa (`sqlite+aiosqlite:///:memory:`, `EmbeddingService(provider="mock")`, `QdrantClient(":memory:")`).
- Commits no padrão Conventional Commits.

---

### Task 1: Dependências e Correção de Contratos de Dados

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/fato_unb/ingestion/models.py`
- Test: `tests/test_rag.py`

**Interfaces:**
- Consumes: `RawDocument` em `src/fato_unb/ingestion/models.py`
- Produces: `RawDocument` com suporte a `str(self.url)` para permitir `HttpUrl` em `set_derived_fields` e dependências `sqlalchemy`, `aiosqlite`, `asyncpg` declaradas no `pyproject.toml`.

- [ ] **Step 1: Escrever teste de unidade reproduzindo suporte a HttpUrl em RawDocument**

Adicionar em `tests/test_ingestion.py`:
```python
from pydantic import HttpUrl
from fato_unb.ingestion.models import RawDocument, SourceType
from datetime import datetime, UTC

def test_raw_document_accepts_http_url():
    doc = RawDocument(
        title="Teste URL",
        content="Conteúdo",
        url=HttpUrl("https://noticias.unb.br/exemplo"),
        source="UnB",
        source_type=SourceType.RSS_NEWS,
        published_at=datetime.now(UTC),
    )
    assert len(doc.doc_id) == 64
    assert str(doc.url) == "https://noticias.unb.br/exemplo"
```

- [ ] **Step 2: Executar teste para verificar falha ou comportamento**

Run: `uv run pytest tests/test_ingestion.py -k "test_raw_document_accepts_http_url"`

- [ ] **Step 3: Ajustar `src/fato_unb/ingestion/models.py` e adicionar dependências ao `pyproject.toml`**

Em `src/fato_unb/ingestion/models.py`:
```python
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
```

Adicionar as dependências no `pyproject.toml` via `uv`:
```bash
uv add "sqlalchemy>=2.0.0" "aiosqlite>=0.20.0" "asyncpg>=0.30.0"
```

- [ ] **Step 4: Executar testes para verificar passagem**

Run: `uv run pytest tests/test_ingestion.py tests/test_rag.py -k "test_chunker"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/fato_unb/ingestion/models.py tests/test_ingestion.py
git commit -m "fix(ingest): handle HttpUrl in RawDocument and declare database dependencies"
```

---

### Task 2: Infraestrutura PostgreSQL no Docker Compose e Configurações de Ambiente

**Files:**
- Modify: `docker-compose.yml`
- Modify: `.env.example`

**Interfaces:**
- Consumes: Configurações de ambiente (`DATABASE_URL`, `QDRANT_HOST`, `QDRANT_PORT`)
- Produces: Serviço PostgreSQL 16 pronto no `docker-compose.yml` e documentação de variáveis em `.env.example`.

- [ ] **Step 1: Atualizar `docker-compose.yml` com serviço PostgreSQL**

Editar `docker-compose.yml` para conter:
```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.19.0
    ports: 
      - "6333:6333"
    volumes:
      - ./qdrant_storage:/qdrant/storage
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: fato_unb
      POSTGRES_USER: fato_user
      POSTGRES_PASSWORD: fato_password
    ports:
      - "5432:5432"
    volumes:
      - ./postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
```

- [ ] **Step 2: Atualizar `.env.example`**

Editar `.env.example` para incluir:
```bash
# Staging Database (PostgreSQL / SQLite fallback)
DATABASE_URL=postgresql+asyncpg://fato_user:fato_password@localhost:5432/fato_unb
# SQLite fallback para desenvolvimento e testes locais sem Docker:
# DATABASE_URL=sqlite+aiosqlite:///./fato_unb.db

# Vector Database (Qdrant)
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

- [ ] **Step 3: Validar sintaxe do `docker-compose.yml`**

Run: `docker compose config` (ou validação de sintaxe YAML)

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml .env.example
git commit -m "feat(infra): add postgresql service to docker-compose and configure database url in env"
```

---

### Task 3: Camada de Repositório de Staging (`StagingRepository`)

**Files:**
- Create: `src/fato_unb/storage/__init__.py`
- Create: `src/fato_unb/storage/db.py`
- Create: `src/fato_unb/storage/repository.py`
- Test: `tests/test_staging_repository.py`

**Interfaces:**
- Consumes: `RawDocument` (`src/fato_unb/ingestion/models.py`)
- Produces: `StagingRepository` com métodos:
  - `save_documents(docs: list[RawDocument]) -> int`
  - `get_pending_documents(limit: int = 100) -> list[RawDocument]`
  - `mark_as_indexed(doc_id: str) -> None`
  - `mark_as_failed(doc_id: str, error: str) -> None`
  - `reset_failed_to_pending() -> int`
  - `get_stats() -> dict[str, int]`

- [ ] **Step 1: Escrever teste de unidade para `StagingRepository`**

Criar `tests/test_staging_repository.py`:
```python
import pytest
from datetime import datetime, UTC
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from fato_unb.ingestion.models import RawDocument, SourceType
from fato_unb.storage.db import Base, IngestionStatus
from fato_unb.storage.repository import StagingRepository

@pytest.fixture
async def test_repo():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    repo = StagingRepository(session_factory=session_factory)
    yield repo
    await engine.dispose()

@pytest.mark.anyio
async def test_save_and_get_pending_documents(test_repo):
    doc1 = RawDocument(
        title="Notícia 1",
        content="Conteúdo 1",
        url="https://noticias.unb.br/1",
        source="UnB",
        source_type=SourceType.RSS_NEWS,
        published_at=datetime.now(UTC),
    )
    doc2 = RawDocument(
        title="Notícia 2",
        content="Conteúdo 2",
        url="https://noticias.unb.br/2",
        source="UnB",
        source_type=SourceType.RSS_NEWS,
        published_at=datetime.now(UTC),
    )
    
    # 1. Salvar documentos
    inserted = await test_repo.save_documents([doc1, doc2])
    assert inserted == 2
    
    # 2. Testar deduplicação: tentar inserir doc1 novamente não insere duplicata
    inserted_duplicate = await test_repo.save_documents([doc1])
    assert inserted_duplicate == 0
    
    # 3. Buscar pendentes
    pending = await test_repo.get_pending_documents(limit=10)
    assert len(pending) == 2
    assert pending[0].doc_id == doc1.doc_id

@pytest.mark.anyio
async def test_mark_as_indexed_and_failed(test_repo):
    doc = RawDocument(
        title="Notícia 3",
        content="Conteúdo 3",
        url="https://noticias.unb.br/3",
        source="UnB",
        source_type=SourceType.RSS_NEWS,
        published_at=datetime.now(UTC),
    )
    await test_repo.save_documents([doc])
    
    # Marcar como INDEXED
    await test_repo.mark_as_indexed(doc.doc_id)
    pending = await test_repo.get_pending_documents()
    assert len(pending) == 0
    
    # Marcar como FAILED
    doc4 = RawDocument(
        title="Notícia 4",
        content="Conteúdo 4",
        url="https://noticias.unb.br/4",
        source="UnB",
        source_type=SourceType.RSS_NEWS,
        published_at=datetime.now(UTC),
    )
    await test_repo.save_documents([doc4])
    await test_repo.mark_as_failed(doc4.doc_id, "Erro no FastEmbed")
    
    stats = await test_repo.get_stats()
    assert stats[IngestionStatus.INDEXED] == 1
    assert stats[IngestionStatus.FAILED] == 1
    assert stats[IngestionStatus.PENDING] == 0
```

- [ ] **Step 2: Executar teste para verificar falha (arquivo ainda não existe)**

Run: `uv run pytest tests/test_staging_repository.py`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/fato_unb/storage/db.py` e `src/fato_unb/storage/repository.py`**

Criar `src/fato_unb/storage/__init__.py`:
```python
from .db import Base, IngestionStatus, RawDocumentEntity, get_session, init_db
from .repository import StagingRepository

__all__ = ["Base", "IngestionStatus", "RawDocumentEntity", "get_session", "init_db", "StagingRepository"]
```

Criar `src/fato_unb/storage/db.py` (movendo e aprimorando o schema de `vectorstore/db.py` com índices e suporte dual):
```python
import enum
import os
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

from dotenv import load_dotenv
from sqlalchemy import DateTime, String, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./fato_unb.db")

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


class IngestionStatus(str, enum.Enum):
    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"


class RawDocumentEntity(Base):
    __tablename__ = "raw_documents"

    doc_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    url: Mapped[str] = mapped_column(
        String(2048), unique=True, index=True, nullable=False
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    semester_ref: Mapped[str] = mapped_column(String(16), nullable=True)

    status: Mapped[IngestionStatus] = mapped_column(
        SAEnum(IngestionStatus), default=IngestionStatus.PENDING, index=True
    )
    collected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str] = mapped_column(Text, nullable=True)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession]:
    async with async_session() as session:
        yield session
```

Criar `src/fato_unb/storage/repository.py`:
```python
from datetime import UTC, datetime
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from fato_unb.ingestion.models import RawDocument, SourceType
from fato_unb.storage.db import IngestionStatus, RawDocumentEntity, async_session


class StagingRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession] | None = None):
        self.session_factory = session_factory or async_session

    async def save_documents(self, docs: list[RawDocument]) -> int:
        if not docs:
            return 0

        inserted_count = 0
        async with self.session_factory() as session:
            async with session.begin():
                for doc in docs:
                    # Verifica existência por doc_id ou url para idempotência
                    query = select(RawDocumentEntity.doc_id).where(
                        (RawDocumentEntity.doc_id == doc.doc_id) | (RawDocumentEntity.url == str(doc.url))
                    )
                    exists = (await session.execute(query)).scalar_one_or_none()
                    if exists:
                        continue

                    entity = RawDocumentEntity(
                        doc_id=doc.doc_id,
                        url=str(doc.url),
                        title=doc.title,
                        content=doc.content,
                        source=doc.source,
                        source_type=doc.source_type.value if hasattr(doc.source_type, 'value') else str(doc.source_type),
                        published_at=doc.published_at,
                        semester_ref=doc.semester_ref,
                        status=IngestionStatus.PENDING,
                        collected_at=datetime.now(UTC),
                    )
                    session.add(entity)
                    inserted_count += 1

        return inserted_count

    async def get_pending_documents(self, limit: int = 100) -> list[RawDocument]:
        async with self.session_factory() as session:
            stmt = (
                select(RawDocumentEntity)
                .where(RawDocumentEntity.status == IngestionStatus.PENDING)
                .order_by(RawDocumentEntity.collected_at.asc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            entities = result.scalars().all()

            docs: list[RawDocument] = []
            for e in entities:
                docs.append(
                    RawDocument(
                        doc_id=e.doc_id,
                        url=e.url,
                        title=e.title,
                        content=e.content,
                        source=e.source,
                        source_type=SourceType(e.source_type),
                        published_at=e.published_at,
                        semester_ref=e.semester_ref,
                    )
                )
            return docs

    async def mark_as_indexed(self, doc_id: str) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                stmt = (
                    update(RawDocumentEntity)
                    .where(RawDocumentEntity.doc_id == doc_id)
                    .values(
                        status=IngestionStatus.INDEXED,
                        indexed_at=datetime.now(UTC),
                        error_message=None,
                    )
                )
                await session.execute(stmt)

    async def mark_as_failed(self, doc_id: str, error: str) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                stmt = (
                    update(RawDocumentEntity)
                    .where(RawDocumentEntity.doc_id == doc_id)
                    .values(
                        status=IngestionStatus.FAILED,
                        error_message=error,
                    )
                )
                await session.execute(stmt)

    async def reset_failed_to_pending(self) -> int:
        async with self.session_factory() as session:
            async with session.begin():
                stmt = (
                    update(RawDocumentEntity)
                    .where(RawDocumentEntity.status == IngestionStatus.FAILED)
                    .values(status=IngestionStatus.PENDING, error_message=None)
                )
                result = await session.execute(stmt)
                return result.rowcount

    async def get_stats(self) -> dict[IngestionStatus, int]:
        async with self.session_factory() as session:
            stmt = select(RawDocumentEntity.status, func.count(RawDocumentEntity.doc_id)).group_by(RawDocumentEntity.status)
            result = await session.execute(stmt)
            counts = {status: 0 for status in IngestionStatus}
            for status, count in result.all():
                counts[status] = count
            return counts
```

- [ ] **Step 4: Executar testes de unidade de StagingRepository**

Run: `uv run pytest tests/test_staging_repository.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fato_unb/storage tests/test_staging_repository.py
git commit -m "feat(storage): implement StagingRepository and SQLAlchemy async models"
```

---

### Task 4: Pipeline Unificado de Indexação (`IndexingPipeline`)

**Files:**
- Create: `src/fato_unb/rag/pipeline.py`
- Test: `tests/test_indexing_pipeline.py`

**Interfaces:**
- Consumes: `StagingRepository`, `SemanticChunker`, `EmbeddingService`, `QdrantClient`
- Produces: `IndexingPipeline` com método assíncrono `run(max_docs: int = 100) -> IndexingReport`

- [ ] **Step 1: Escrever teste de unidade com Mock Embedder e In-Memory Qdrant**

Criar `tests/test_indexing_pipeline.py`:
```python
import pytest
from datetime import datetime, UTC
from qdrant_client import QdrantClient
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from fato_unb.ingestion.models import RawDocument, SourceType
from fato_unb.rag.chunker import SemanticChunker
from fato_unb.rag.embeddings import EmbeddingService
from fato_unb.rag.pipeline import IndexingPipeline
from fato_unb.storage.db import Base, IngestionStatus
from fato_unb.storage.repository import StagingRepository


@pytest.fixture
async def setup_pipeline():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    repo = StagingRepository(session_factory=session_factory)
    qdrant = QdrantClient(":memory:")
    embedder = EmbeddingService(provider="mock", mock_dimension=384)
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=10)
    
    pipeline = IndexingPipeline(
        repository=repo,
        chunker=chunker,
        embedder=embedder,
        qdrant_client=qdrant,
        collection_name="test_fato_noticias",
        batch_size=32
    )
    
    yield pipeline, repo, qdrant
    await engine.dispose()


@pytest.mark.anyio
async def test_indexing_pipeline_end_to_end(setup_pipeline):
    pipeline, repo, qdrant = setup_pipeline
    
    doc = RawDocument(
        title="Comunicado da Reitoria",
        content="As aulas do primeiro semestre iniciam regularmente em todas as unidades acadêmicas.",
        url="https://unb.br/comunicado-reitoria",
        source="UnB",
        source_type=SourceType.RSS_NEWS,
        published_at=datetime.now(UTC),
        semester_ref="2026.1"
    )
    await repo.save_documents([doc])
    
    # Executa o pipeline
    report = await pipeline.run(max_docs=10)
    
    assert report.total_processed == 1
    assert report.total_indexed == 1
    assert report.total_chunks >= 1
    assert report.total_failed == 0
    
    # Verifica que o documento no repo está como INDEXED
    stats = await repo.get_stats()
    assert stats[IngestionStatus.INDEXED] == 1
    assert stats[IngestionStatus.PENDING] == 0
    
    # Verifica que o Qdrant recebeu os pontos
    points_count = qdrant.count(collection_name="test_fato_noticias").count
    assert points_count == report.total_chunks


@pytest.mark.anyio
async def test_indexing_pipeline_handles_empty_document(setup_pipeline):
    pipeline, repo, qdrant = setup_pipeline
    
    empty_doc = RawDocument(
        title="Documento Sem Conteúdo",
        content="   ",
        url="https://unb.br/vazio",
        source="UnB",
        source_type=SourceType.HTML_PAGE,
        published_at=datetime.now(UTC),
    )
    await repo.save_documents([empty_doc])
    
    report = await pipeline.run(max_docs=10)
    assert report.total_processed == 1
    assert report.total_failed == 1
    
    stats = await repo.get_stats()
    assert stats[IngestionStatus.FAILED] == 1
    assert stats[IngestionStatus.INDEXED] == 0
```

- [ ] **Step 2: Executar teste para verificar falha (pipeline.py ainda não existe)**

Run: `uv run pytest tests/test_indexing_pipeline.py`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/fato_unb/rag/pipeline.py`**

Criar `src/fato_unb/rag/pipeline.py`:
```python
import logging
import uuid
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, SparseVectorParams

from fato_unb.rag.chunker import SemanticChunker
from fato_unb.rag.embeddings import EmbeddingService
from fato_unb.storage.repository import StagingRepository
from fato_unb.vectorstore.client import get_qdrant_client

logger = logging.getLogger(__name__)
NAMESPACE = uuid.UUID("f77e4b30-9222-4d60-889a-861c4da36012")


@dataclass
class IndexingReport:
    total_processed: int = 0
    total_indexed: int = 0
    total_chunks: int = 0
    total_failed: int = 0


class IndexingPipeline:
    def __init__(
        self,
        repository: StagingRepository | None = None,
        chunker: SemanticChunker | None = None,
        embedder: EmbeddingService | None = None,
        qdrant_client: QdrantClient | None = None,
        collection_name: str = "fato_unb_noticias",
        batch_size: int = 64,
    ):
        self.repository = repository or StagingRepository()
        self.chunker = chunker or SemanticChunker()
        self.embedder = embedder or EmbeddingService(provider="local")
        self.client = qdrant_client or get_qdrant_client()
        self.collection_name = collection_name
        self.batch_size = batch_size

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.embedder.vector_dimension,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={"sparse": SparseVectorParams()}
            )
            logger.info(f"Coleção Qdrant '{self.collection_name}' criada com sucesso.")

    async def run(self, max_docs: int = 100) -> IndexingReport:
        report = IndexingReport()
        self.ensure_collection()

        docs = await self.repository.get_pending_documents(limit=max_docs)
        if not docs:
            logger.info("Nenhum documento pendente para indexação.")
            return report

        logger.info(f"Iniciando pipeline de indexação para {len(docs)} documentos pendentes.")

        for doc in docs:
            report.total_processed += 1
            try:
                chunks = self.chunker.chunk_document(doc)
                if not chunks:
                    logger.warning(f"Documento {doc.doc_id} ({doc.title}) não gerou chunks.")
                    await self.repository.mark_as_failed(doc.doc_id, "Corpo de texto vazio ou não particionável")
                    report.total_failed += 1
                    continue

                # Processa chunks em lotes de embedding
                for i in range(0, len(chunks), self.batch_size):
                    chunk_batch = chunks[i : i + self.batch_size]
                    texts = [c.content for c in chunk_batch]
                    vectors = self.embedder.embed_texts(texts)

                    points = [
                        PointStruct(
                            id=str(uuid.uuid5(NAMESPACE, chunk.chunk_id)),
                            vector={"dense": vec},
                            payload=chunk.model_dump(mode="json"),
                        )
                        for chunk, vec in zip(chunk_batch, vectors)
                    ]

                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                        wait=True,
                    )
                    report.total_chunks += len(points)

                await self.repository.mark_as_indexed(doc.doc_id)
                report.total_indexed += 1

            except Exception as exc:
                logger.error(f"Erro ao indexar documento {doc.doc_id}: {exc}", exc_info=True)
                await self.repository.mark_as_failed(doc.doc_id, str(exc))
                report.total_failed += 1

        logger.info(f"Indexação finalizada: {report}")
        return report
```

- [ ] **Step 4: Executar testes de unidade de IndexingPipeline**

Run: `uv run pytest tests/test_indexing_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fato_unb/rag/pipeline.py tests/test_indexing_pipeline.py
git commit -m "feat(rag): implement unified IndexingPipeline with Qdrant and StagingRepository"
```

---

### Task 5: Integração da Ingestão com Staging e Atualização do Scheduler

**Files:**
- Modify: `src/fato_unb/ingestion/crawler.py`
- Modify: `src/fato_unb/ingestion/rss.py`
- Modify: `src/fato_unb/ingestion/scheduler.py`
- Modify: `demo.py`

**Interfaces:**
- Consumes: `fetch_unb_rss_feed`, `crawl_unb_site`, `StagingRepository`, `IndexingPipeline`
- Produces: `pipeline_job` que executa Ingestão para Staging seguido de Indexação Vetorial, mantendo compatibilidade com persistência em arquivo opcional.

- [ ] **Step 1: Adaptar `src/fato_unb/ingestion/scheduler.py` para utilizar o Staging e o Pipeline**

Modificar `src/fato_unb/ingestion/scheduler.py`:
- Importar `init_db` e `StagingRepository` de `fato_unb.storage`.
- Importar `IndexingPipeline` de `fato_unb.rag.pipeline`.
- Atualizar `run_rss_ingestion` e `run_crawler` para persistir diretamente no `StagingRepository`.
- No `pipeline_job`: inicializar o banco (`await init_db()`), coletar notícias salvando no banco com status `PENDING`, e em seguida chamar `IndexingPipeline.run()`.

- [ ] **Step 2: Atualizar `demo.py` para demonstrar o pipeline ponta a ponta**

Atualizar `demo.py`:
```python
import asyncio
import logging
from fato_unb.storage.db import init_db
from fato_unb.storage.repository import StagingRepository
from fato_unb.ingestion.scheduler import pipeline_job

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("demo_pipeline")

async def run_demo():
    logger.info("Iniciando demonstração do Pipeline Unificado (Ingestão -> Staging -> Chunks -> Embeddings -> Qdrant)...")
    await init_db()
    await pipeline_job()
    
    repo = StagingRepository()
    stats = await repo.get_stats()
    logger.info(f"Estatísticas finais do Staging: {stats}")

if __name__ == "__main__":
    asyncio.run(run_demo())
```

- [ ] **Step 3: Testar execução do scheduler job com SQLite em memória ou arquivo local**

Run: `uv run python demo.py`
Expected: Pipeline executa ciclo de ingestão, indexação e imprime as estatísticas finais.

- [ ] **Step 4: Commit**

```bash
git add src/fato_unb/ingestion/scheduler.py demo.py
git commit -m "feat(pipeline): connect ingestion to staging repository and orchestrate indexing in scheduler"
```

---

### Task 6: Validação de Regressão e Cobertura Completa de Testes

**Files:**
- Test: `tests/`
- Documentation: `README.md`

**Interfaces:**
- Consumes: Todo o pipeline construído
- Produces: Suíte de testes automatizados verde e documentação atualizada no README.

- [ ] **Step 1: Executar suite completa de testes unitários**

Run: `uv run pytest tests/test_ingestion.py tests/test_rag.py tests/test_staging_repository.py tests/test_indexing_pipeline.py -v`
Expected: Todos os testes unitários passando.

- [ ] **Step 2: Atualizar README.md com instruções do Docker Compose e Banco de Dados**

Adicionar seção no `README.md` documentando como subir PostgreSQL e Qdrant com `docker compose up -d` e como executar o pipeline unificado com `uv run python demo.py`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document unified pipeline and database setup in README"
```
