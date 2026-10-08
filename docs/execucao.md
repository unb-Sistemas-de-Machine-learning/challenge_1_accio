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

## 8. Criar o testset e avaliar com RAGAS

Esta avaliação usa `dados.txt` para gerar o testset e depois testa as afirmações
contra o verificador do bot. Execute os comandos a partir da raiz do repositório.

!!! warning "Memória"
    RAM estimada para os modelos: `qwen2.5:7b` **4,7 GB**; `qwen2.5:3b`
    **1,9 GB**; `bge-m3` **1,2 GB**; os três juntos **7,8 GB**. A RAM real
    pode ser maior conforme o contexto e o uso do sistema. Reserve memória
    adicional para o Docker, que também executa Qdrant e PostgreSQL; o consumo
    varia conforme o tamanho do corpus e do índice, e não há um limite fixo
    configurado no `docker-compose.yml`.

### Preparar dependências e modelos

```bash
uv sync --extra evaluation --dev
ollama pull qwen2.5:7b
ollama pull qwen2.5:3b
ollama pull bge-m3
```

Com o Ollama em execução, gere o testset:

```bash
uv run python -m fato_unb.evaluation.generateTestset
```

O comando cria ou substitui `ragas_testset_local.csv` na raiz do projeto.

### Avaliar o bot

O modo `--bot` consulta o Qdrant e usa `GEMINI_API_KEY` do `.env`. Para testar
uma linha:

```bash
uv run python -m fato_unb.evaluation.evaluator --bot --limit 1 --output ./resultados_ragas.csv
```

Para avaliar o testset inteiro, omita `--limit`:

```bash
uv run python -m fato_unb.evaluation.evaluator --bot --output ./resultados_ragas.csv
```

Para continuar uma execução interrompida, mantenha o mesmo testset e arquivo de
saída; aumente o limite e acrescente `--resume`:

```bash
uv run python -m fato_unb.evaluation.evaluator --bot --limit 5 --output ./resultados_ragas.csv
uv run python -m fato_unb.evaluation.evaluator --bot --limit 10 --output ./resultados_ragas.csv --resume
```

São gerados `resultados_ragas.csv` com as respostas e métricas,
`resultados_ragas_resumo.json` com as médias e
`resultados_ragas.csv.checkpoint.json` para retomada. Não gere novamente o
testset antes de retomar a mesma avaliação.

## 9. Documentação

```bash
uv run mkdocs serve           # abre em http://127.0.0.1:8000
uv run mkdocs build --strict  # o mesmo build do CI
```

O site é publicado no GitHub Pages por `.github/workflows/ci.yml`, a cada push na branch `develop`.
