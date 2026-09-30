# Fato UnB - Bot checador de fatos da UnB no Telegram

Challenge 1 - Equipe Accio - Sistemas de Machine Learning 2026/02

Pipeline unificado de ingestão contínua (Web Crawler & RSS Feeds), persistência em banco relacional de staging e indexação vetorial (Qdrant) para recuperação aumentada de informações (RAG).

---

## 🚀 Pré-requisitos

- [uv](https://docs.astral.sh/uv/) (gerenciador de dependências e ambiente Python)
- [Docker](https://www.docker.com/) e [Docker Compose](https://docs.docker.com/compose/) (para infraestrutura de banco e busca vetorial)

### Instalação do `uv`

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## ⚙️ Configuração de Ambiente e Variáveis

Crie o arquivo `.env` a partir do modelo fornecido:

```bash
cp .env.example .env
```

### Modos de Banco de Dados:
- **Produção / Docker (PostgreSQL 16)** (padrão):
  ```env
  DATABASE_URL=postgresql+asyncpg://fato_user:fato_password@localhost:5432/fato_unb
  QDRANT_HOST=localhost
  QDRANT_PORT=6333
  ```
- **Desenvolvimento / Testes Locais (SQLite fallback)**:
  Caso queira executar sem Docker para o banco relacional, utilize a URL do SQLite no `.env`:
  ```env
  DATABASE_URL=sqlite+aiosqlite:///./fato_unb.db
  ```

---

## 🐳 Infraestrutura com Docker Compose

Suba os serviços do PostgreSQL 16 e Qdrant 1.19 em segundo plano:

```bash
docker compose up -d
```

Para verificar o status dos contêineres:
```bash
docker compose ps
```

Para encerrar os serviços:
```bash
docker compose down
```

---

## 🏃 Como Executar

### 1. Instalar Dependências
```bash
uv sync
```

### 2. Executar o Pipeline Unificado de Demonstração
Executa um ciclo completo de ponta a ponta: Ingestão (Crawler e RSS) -> Persistência no Staging -> Geração de Chunks e Embeddings -> Indexação no Qdrant:

```bash
uv run python demo.py
```

### 3. Buscar evidências para uma alegação
Consulta a coleção indexada usando o mesmo modelo de embedding do índice (busca híbrida denso + BM25 e, se configurado, reranker):

```bash
uv run python demo_busca.py "A UnB vai cobrar mensalidade?" -l 3 -c
```

Em código (é a interface que o bot do Telegram deve usar):

```python
from fato_unb.rag.retriever import Retriever

retriever = Retriever.from_env()          # lê EMBEDDING_MODEL e RERANKER_MODEL
for ev in retriever.buscar("O RU aceita a carteirinha antiga?", limit=3):
    print(ev.title, ev.url, ev.score)      # ev.contexto = trecho ao redor, para o LLM
```

### 4. Trocar o modelo ou o chunker (reindexar)
Vetores de modelos diferentes não podem conviver na mesma coleção: cada modelo tem a sua
(`fato_unb_noticias__multilingual-e5-large` para o padrão atual; `fato_unb_noticias` guarda os dados do MiniLM antigo). Após mudar
`EMBEDDING_MODEL`, o chunker ou a configuração do índice, reindexe todo o staging:

```bash
uv run python scripts/reindexar.py        # pede confirmação antes de recriar a coleção
```

Variáveis (veja `.env.example`): `EMBEDDING_MODEL`, `RERANKER_MODEL` (vazio = sem reranker) e
`FASTEMBED_CACHE_PATH`. **Se o seu `/tmp` for tmpfs (RAM), defina `FASTEMBED_CACHE_PATH` para um
diretório em disco**: modelos como o e5-large (2,2 GB) enchem a memória e o sistema mata o processo.

### 5. Avaliar a recuperação
```bash
uv run python scripts/avaliar.py --config baseline chunk120-idf e5-c120
```
Métricas por documento, por chunk e por contexto entregue ao LLM em
`docs/modulos/avaliacao/dataset-teste.md`.

### 6. Executar o Agendador Contínuo (Scheduler)
Inicia o processo em background que monitora e roda o ciclo de ingestão e indexação a cada 1 hora:

```bash
uv run python -m fato_unb.ingestion.scheduler
```

---

## 🧪 Testes Automatizados

Para rodar a suíte completa de testes unitários e de integração:

```bash
uv run pytest
```

Para rodar com saída detalhada:
```bash
uv run pytest -v
```

---

## 🛠️ Desenvolvimento e Contribuição

Consulte o nosso guia de configuração do ambiente e padrões em [CONTRIBUTING.md](CONTRIBUTING.md).
