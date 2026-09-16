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

### 3. Executar o Agendador Contínuo (Scheduler)
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
