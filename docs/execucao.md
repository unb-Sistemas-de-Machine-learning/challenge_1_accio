# Como Executar

## Você vai precisar de

- [uv](https://docs.astral.sh/uv/) (instala o Python ≥ 3.14 e as dependências)
- Docker e Docker Compose (para o Qdrant e o PostgreSQL)
- Chave do Gemini ([AI Studio](https://aistudio.google.com/apikey)) e, para o bot, um token do [@BotFather](https://t.me/BotFather)

## 1. Instalar

```bash
git clone https://github.com/unb-Sistemas-de-Machine-learning/challenge_1_accio.git
cd challenge_1_accio
uv sync --all-extras --dev
uv pip install -e .
```

## 2. Configurar

```bash
cp .env.example .env
```

Preencha no `.env` pelo menos `GEMINI_API_KEY` e, para o bot, `TELEGRAM_BOT_TOKEN`. As demais variáveis estão comentadas no próprio `.env.example`.

!!! warning "Memória"
    O modelo de embedding padrão usa cerca de 2,2 GB de memória. Se o seu `/tmp` fica na RAM, defina `FASTEMBED_CACHE_PATH` para uma pasta em disco.

## 3. Subir os bancos

```bash
docker compose up -d
```

O Qdrant fica em `localhost:6333` (painel em <http://localhost:6333/dashboard>) e o PostgreSQL em `localhost:5432`.

## 4. Coletar e indexar

```bash
uv run python -m fato_unb.ingestion.scheduler
```

Roda um ciclo agora e repete **a cada 1 hora**: coleta, grava no staging, divide em trechos e indexa no Qdrant. Se trocar o modelo de embedding ou o chunker, use `uv run python scripts/reindexar.py`.

## 5. Checar uma afirmação

```bash
uv run python scripts/checar.py "As inscrições do PAS 1 vão até 30 de setembro"
uv run python scripts/checar.py "..." --dry-run   # mostra o prompt, sem chamar o LLM
```

## 6. Rodar o bot

```bash
uv run python -m fato_unb.bots.telegram_bot
```

## 7. Testes

```bash
uv run pytest tests/ -v
```

Os testes de `tests/test_vectorstore.py` que usam o Qdrant real precisam do container no ar. Os demais não precisam de Docker.

## 8. Avaliação

```bash
uv run python scripts/avaliar.py --config baseline --falhas   # busca
uv run python scripts/avaliar_vereditos.py --falhas           # vereditos (usa o LLM)
```

## 9. Documentação

```bash
uv run mkdocs serve           # abre em http://127.0.0.1:8000
uv run mkdocs build --strict  # o mesmo build do CI
```

O site é publicado no GitHub Pages por `.github/workflows/ci.yml`, a cada push na branch `develop`.
