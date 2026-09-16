# Especificação Técnica: Pipeline Unificado de Ingestão e Indexação Vetorial

- **Data de Criação:** 2026-09-16
- **Status:** Aprovado em Design
- **Módulos Afetados:** `src/fato_unb/storage/`, `src/fato_unb/rag/`, `src/fato_unb/ingestion/`, `docker-compose.yml`, `pyproject.toml`
- **Conformidade Constitucional:** Atende aos Princípios I (Soberania da Evidência), III (Desacoplamento e Idempotência) e V (Contratos Estritos) da [Constituição FatoUnB](../../.specify/memory/constitution.md).

---

## 1. Contexto e Motivação

Atualmente, o projeto FatoUnB possui componentes isolados para coleta de dados (`crawler.py`, `rss.py`), particionamento de texto (`chunker.py`), cálculo de vetores (`embeddings.py`) e persistência vetorial (`operations.py`, `collections.py`). Além disso, foi modelada uma entidade relacional de staging (`RawDocumentEntity`), mas ela ainda não estava conectada ao ciclo de vida da indexação.

Esta especificação formaliza a integração desses subsistemas em um **Pipeline Unificado e Desacoplado de Duas Etapas**:
1. **Etapa 1 (Ingestão & Staging Relacional):** Coleta documentos oficiais (Crawler e RSS) e persiste no PostgreSQL/SQLite com status inicial `PENDING` e controle de unicidade por hash determinístico (`doc_id`).
2. **Etapa 2 (Indexação Vetorial):** Consome documentos `PENDING` do banco relacional, executa chunking semântico com injeção de contexto, gera embeddings densos de 384 dimensões em lotes, faz o upsert idempotente no Qdrant e atualiza o status de cada documento para `INDEXED` (ou `FAILED` com mensagem de erro).

---

## 2. Arquitetura do Sistema

```
                      ┌─────────────────────────────────┐
                      │    Fontes Oficiais UnB          │
                      │  (Notícias, RSS, Editais, HTML) │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────┐
                      │  Módulo de Ingestão             │
                      │  - crawler.py & rss.py          │
                      └────────────────┬────────────────┘
                                       │ RawDocument (Pydantic)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Camada de Staging Relacional (PostgreSQL / SQLite fallback)                │
│  Tabela: raw_documents                                                      │
│  - doc_id (PK, SHA256 da URL)                                               │
│  - url (Unique), title, content, source, source_type                        │
│  - published_at, semester_ref                                               │
│  - status: PENDING | INDEXED | FAILED                                        │
│  - collected_at, indexed_at, error_message                                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Documentos PENDING
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  IndexingPipeline (Pipeline Unificado)                                      │
│                                                                             │
│  1. StagingRepository.get_pending_documents(limit=100)                      │
│     │                                                                       │
│     ▼                                                                       │
│  2. SemanticChunker.chunk_document(doc)                                     │
│     └──> Gera DocumentChunk com prefixo contextual enriquecido              │
│     │                                                                       │
│     ▼                                                                       │
│  3. EmbeddingService.embed_texts(batch_texts)                               │
│     └──> Gera vetores densos (384 dimensões, FastEmbed ONNX)                │
│     │                                                                       │
│     ▼                                                                       │
│  4. Qdrant Client Upsert                                                    │
│     └──> PointStruct(id=uuid5(NAMESPACE, chunk_id), vector, payload)        │
│     │                                                                       │
│     ▼                                                                       │
│  5. StagingRepository.mark_as_indexed(doc_id) / mark_as_failed(doc_id, err)  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Repositório Vetorial (Qdrant)                                              │
│  Collection: 'fato_unb_noticias'                                            │
│  - Vetor denso: 384 dimensões (Cosseno)                                     │
│  - Payload completo: title, url, source, semester_ref, published_at, etc.   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Contratos de Dados e Modelos

### 3.1. Modelo de Domínio: `RawDocument` (Pydantic v2)
Localizado em `src/fato_unb/ingestion/models.py`.
```python
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
    semester_ref: str | None = None
    doc_id: str = Field(default="")
```
*Garantia:* `doc_id` gerado via `sha256(url.encode('utf-8')).hexdigest()`.

### 3.2. Modelo Relacional: `RawDocumentEntity` (SQLAlchemy 2.0 Async)
Localizado em `src/fato_unb/storage/db.py` (ou `vectorstore/db.py`).
- Tabela: `raw_documents`
- Campos:
  - `doc_id: str` (PK, 64 chars)
  - `url: str` (Unique, indexado)
  - `title: str`
  - `content: str` (Text)
  - `source: str` (Indexado)
  - `source_type: str`
  - `published_at: datetime | None` (Timezone-aware)
  - `semester_ref: str | None` (ex: "2026.1")
  - `status: IngestionStatus` (Enum: `PENDING`, `INDEXED`, `FAILED`, indexado)
  - `collected_at: datetime` (Default: UTC now)
  - `indexed_at: datetime | None`
  - `error_message: str | None`

### 3.3. Modelo de Fragmento: `DocumentChunk` (Pydantic v2)
Localizado em `src/fato_unb/rag/models.py`.
- `chunk_id: str` (Hash determinístico de 16 caracteres: `sha256(f"{doc_id}_{idx}").hexdigest()[:16]`)
- `doc_id: str`
- `content: str` (Texto com injeção contextual: `[Documento: {title}]\n[Fonte: {source} | Ref: {ref}]\n\n{raw_text}`)
- `raw_text: str`
- `chunk_index: int`
- `total_chunks: int`
- Metadados de filtro: `title`, `url`, `source`, `semester_ref`, `published_at`

### 3.4. Ponto Vetorial no Qdrant
- `id`: UUID RFC-4122 determinístico: `uuid.uuid5(NAMESPACE, chunk.chunk_id)`
- `vector`: `{"dense": list[float]}` (384 dimensões)
- `payload`: dicionário JSON gerado por `chunk.model_dump(mode="json")`

---

## 4. Componentes Detalhados

### 4.1. Camada de Repositório: `StagingRepository`
Interface desacoplada que gerencia transações com a base de dados relacional:

- `async def save_documents(self, docs: list[RawDocument]) -> int`:
  Insere documentos no banco. Em caso de colisão de chave primária (`doc_id` ou `url`), ignora a inserção sem lançar erro, preservando registros existentes. Retorna o total de novos documentos inseridos.
- `async def get_pending_documents(self, limit: int = 100) -> list[RawDocument]`:
  Recupera até `limit` registros com `status == IngestionStatus.PENDING`, ordenados por `collected_at ASC`.
- `async def mark_as_indexed(self, doc_id: str) -> None`:
  Atualiza `status = IngestionStatus.INDEXED`, `indexed_at = datetime.now(UTC)` e limpa `error_message`.
- `async def mark_as_failed(self, doc_id: str, error: str) -> None`:
  Atualiza `status = IngestionStatus.FAILED` e registra o erro em `error_message`.
- `async def reset_failed_to_pending(self) -> int`:
  Permite re-tentar documentos com falha, alterando status de `FAILED` para `PENDING`.

### 4.2. Pipeline Unificado: `IndexingPipeline`
Orquestrador central de indexação vetorial:

- **Dependências Injetáveis:**
  - `repository: StagingRepository`
  - `chunker: SemanticChunker`
  - `embedder: EmbeddingService`
  - `qdrant_client: QdrantClient`
  - `collection_name: str` (padrão: `"fato_unb_noticias"`)
  - `batch_size: int` (padrão: 64 chunks)

- **Fluxo do Método `async def run(self, max_docs: int = 100) -> IndexingReport`:**
  1. Cria coleção no Qdrant se ela não existir (`create_collection`).
  2. Obtém lote de documentos pendentes via `repository.get_pending_documents(limit=max_docs)`.
  3. Se a lista estiver vazia, encerra a execução retornando relatório vazio.
  4. Para cada documento:
     - Tenta fatiar com `chunker.chunk_document(doc)`.
     - Se o documento não gerar chunks válidos (ex: corpo vazio), chama `repository.mark_as_failed(doc.doc_id, "Corpo de texto vazio ou não particionável")`.
     - Se gerar chunks, adiciona à fila de lotes a serem indexados.
  5. Agrupa os chunks em lotes de tamanho `batch_size`:
     - Gera embeddings densos com `embedder.embed_texts(...)`.
     - Monta `PointStruct` com UUID determinístico e payload serializado.
     - Executa `qdrant_client.upsert(collection_name, points=points, wait=True)`.
  6. Para cada documento cujos chunks foram inseridos com sucesso:
     - Chama `repository.mark_as_indexed(doc.doc_id)`.
  7. Se ocorrer uma falha crítica durante o lote de upsert/embedding, captura a exceção e marca os documentos afetados como `FAILED`.
  8. Retorna métricas consolidadas em `IndexingReport` (`total_processed`, `total_indexed`, `total_chunks`, `total_failed`).

### 4.3. Integração com o Agendador (`scheduler.py`)
O job periódico do agendador é atualizado para executar de forma sequencial e robusta:
```python
async def pipeline_job():
    logger.info("Iniciando ciclo de ingestão...")
    await run_crawler_to_db()
    await run_rss_to_db()
    
    logger.info("Iniciando ciclo de indexação vetorial...")
    pipeline = IndexingPipeline()
    report = await pipeline.run()
    logger.info(f"Ciclo concluído: {report}")
```

---

## 5. Infraestrutura e Variáveis de Ambiente

### 5.1. `docker-compose.yml`
Inclusão do serviço de PostgreSQL 16 junto ao Qdrant:
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

### 5.2. Variáveis de Ambiente (`.env.example`)
```bash
# Banco Relacional (Staging)
# Produção / Docker Compose:
DATABASE_URL=postgresql+asyncpg://fato_user:fato_password@localhost:5432/fato_unb
# Desenvolvimento / Testes locais sem Docker:
# DATABASE_URL=sqlite+aiosqlite:///./fato_unb.db

# Banco Vetorial (Qdrant)
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

---

## 6. Estratégia de Testes e Validação

1. **Testes Unitários do Repositório (`tests/test_staging_repository.py`):**
   - Utiliza banco SQLite assíncrono em memória (`sqlite+aiosqlite:///:memory:`).
   - Testa: salvamento com deduplicação, consulta de documentos pendentes, atualização para `INDEXED` e marcação de `FAILED`.

2. **Testes Unitários do Pipeline (`tests/test_indexing_pipeline.py`):**
   - Utiliza `EmbeddingService(provider="mock")` (sem download de modelos externos).
   - Utiliza Qdrant em memória (`QdrantClient(":memory:")`).
   - Testa:
     - Fluxo normal de ponta a ponta (`PENDING` -> chunks -> Qdrant -> `INDEXED`).
     - Idempotência de re-indexação.
     - Documentos com texto corrompido ou vazio recebem status `FAILED` sem derrubar a execução.

3. **Testes de Integração (`tests/test_pipeline_integration.py`):**
   - Marcados com `@pytest.mark.integration`.
   - Executa teste com embeddings reais locais e conexões reais caso configuradas.

---

## 7. Critérios de Sucesso e Verificação

- [ ] Todos os testes unitários (`pytest`) passam com 100% de sucesso.
- [ ] O pipeline processa e indexa documentos sem duplicidade no Qdrant e no PostgreSQL.
- [ ] Repetir a execução com os mesmos dados resulta em 0 novos documentos gravados (idempotência comprovada).
- [ ] Documentos com erro são marcados no banco com status `FAILED` e mensagem auditável.
- [ ] Execução compatível tanto com SQLite local quanto com PostgreSQL via Docker.
