FROM python:3.14-slim

# Copia o binário do uv da imagem oficial (jeito recomendado, sem precisar baixar nada em runtime)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependências do sistema que o pymupdf (extração de PDF) precisa para compilar/rodar
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia só os arquivos de dependência primeiro: cacheia essa camada enquanto o código muda
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Agora copia o resto do projeto e instala o pacote em si (uv sync não registra o
# pacote local sozinho, por isso o "pip install -e ." logo depois)
COPY . .
RUN uv sync --frozen && uv pip install -e .

CMD ["uv", "run", "python", "-m", "fato_unb.bots.telegram_bot"]
