# Banco Vetorial e Staging

**Responsável:** Ângelo Araujo Cordova · **Código:** `src/fato_unb/vectorstore/` e `src/fato_unb/storage/`

O armazenamento é o ponto de encontro dos dois fluxos: a coleta **escreve**, a checagem **lê**. São dois bancos:

| Banco | Para quê | O que guarda |
|---|---|---|
| **PostgreSQL** (staging) | Fila de documentos a indexar | O documento bruto e o status: `pending`, `indexed` ou `failed` |
| **Qdrant** | Busca por significado e por palavras | Um ponto por trecho (chunk): vetores + metadados |

## Qdrant no Docker

```yaml
qdrant:
  image: qdrant/qdrant:v1.19.0
  ports: ["6333:6333"]
  volumes: ["./qdrant_storage:/qdrant/storage"]
```

- **Versão fixa em vez de `latest`:** uma atualização automática poderia mudar o formato dos dados e quebrar o que já foi indexado.
- **Volume local:** os vetores continuam lá depois de `docker compose down`.
- O cliente (`client.py`) lê `QDRANT_HOST` e `QDRANT_PORT` do `.env` e usa `@cache`, para abrir uma única conexão.

## Coleção (`collections.py`)

Cada ponto tem dois vetores, **com nome**:

| Vetor | Tipo | Serve para |
|---|---|---|
| `dense` | Denso, distância Cosine | Achar trechos com o mesmo significado |
| `sparse` | Esparso (BM25) | Achar palavras exatas: siglas, nomes, números |

- **Vetores com nome:** uma coleção criada com vetor sem nome não aceita outro vetor depois sem ser recriada.
- **Slot esparso declarado desde o início**, para a busca híbrida chegar sem recriar a coleção.
- **Tamanho do vetor lido do modelo** (`embedder.vector_dimension`), nunca escrito à mão. Trocar de modelo não exige mexer no código.
- **IDF ligado no índice esparso:** sem ele, palavras raras pesam o mesmo que palavras comuns.
- **Uma coleção por modelo** (`collection_name_for`): vetores de modelos diferentes não podem ficar juntos.
- **`ensure_collection` pode rodar quantas vezes quiser:** cria se não existir e, se existir com tamanho diferente, avisa com erro claro.

## Upsert sem duplicar (`operations.py`)

`upsert_documents` calcula os vetores de cada chunk e grava no Qdrant.

- **ID do ponto:** `uuid.uuid5(NAMESPACE, chunk_id)`. O Qdrant só aceita número inteiro ou UUID, e o `chunk_id` é um hash em texto. O `uuid5` converte o hash num UUID **sempre igual para a mesma entrada**.
- **Resultado:** indexar o mesmo documento de novo **sobrescreve** os pontos, não duplica.
- **Payload:** `chunk.model_dump(mode="json")`. O `mode="json"` converte `HttpUrl` e `datetime` em texto, que o Qdrant aceita.
- **Filtros dependem do payload:** `source`, `semester_ref` e `published_at` precisam estar gravados desde o upsert para poderem filtrar depois.

## Busca com filtros (`buscar`)

```python
buscar(query, embedder, collection_name=...,
       source="dpg.unb.br", semester_ref="2026.2",
       data_inicio=..., data_fim=..., limit=5)
```

1. Expande siglas da UnB antes de buscar (ex.: `RU` → "restaurante universitário").
2. Busca nos dois vetores com os filtros aplicados **antes** do ranking.
3. Junta as duas listas com **RRF** (Reciprocal Rank Fusion, que combina pela posição).
4. Se a coleção não tiver o vetor esparso, usa só o denso.
5. Se houver reranker, ele reordena os 20 melhores candidatos.

## Staging (`storage/`)

A tabela `raw_documents` guarda cada `RawDocument` com seu status. Ela existe para:

- **Separar coleta de indexação:** se a indexação falhar, a coleta não se perde.
- **Reindexar sem coletar de novo:** `reset_all_to_pending()` + `scripts/reindexar.py`.
- **Isolar falhas:** um documento com erro fica `failed` com a mensagem, sem travar o lote.

## Testes

`tests/test_vectorstore.py` (criação da coleção, upsert e busca híbrida, fallback para busca densa), `test_staging_repository.py` e `test_indexing_pipeline.py`. Dois testes de `test_vectorstore.py` precisam do Qdrant rodando.

## Limitações

- Ainda não criamos índices de payload (`create_payload_index`) nos campos de filtro.
- O mesmo portal aparece como `noticias.unb.br` (crawler) e `UnB Notícias` (RSS), o que atrapalha o filtro por `source`.
- `upsert_documents` e `buscar` ainda têm a coleção antiga (`fato_unb_noticias`) como padrão; o pipeline e o `Retriever` passam o nome certo.
