#!/usr/bin/env bash
# Prepara uma VM Ubuntu (x86 ou ARM) para rodar o FatoUnB com docker compose.
# Uso, na VM, dentro da pasta do repositório:  bash scripts/setup_vm.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v docker >/dev/null 2>&1; then
    echo ">> Instalando Docker..."
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER"
    NEEDS_RELOGIN=1
fi

# O compose monta ./dados.txt dentro do container; se o arquivo não existir, o Docker
# cria um diretório com esse nome e o crawler quebra.
if [ -d dados.txt ]; then
    echo "ERRO: ./dados.txt é um diretório (criado por um 'up' anterior). Remova-o: rmdir dados.txt" >&2
    exit 1
fi
touch dados.txt

if [ ! -f .env ]; then
    echo ">> Criando .env a partir do .env.example..."
    cp .env.example .env
    # Hex evita caracteres que quebrariam a DATABASE_URL (@, :, /, %).
    SENHA="$(openssl rand -hex 16)"
    sed -i "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=${SENHA}|" .env
    chmod 600 .env
    echo ">> Edite o .env e preencha TELEGRAM_BOT_TOKEN e a chave do LLM (GEMINI_API_KEY ou ANTHROPIC_API_KEY)."
else
    chmod 600 .env
fi

if [ "${NEEDS_RELOGIN:-0}" = "1" ]; then
    echo ">> Docker instalado. Saia e entre de novo na sessão SSH (para o grupo 'docker' valer) e rode:"
else
    echo ">> Pronto. Depois de preencher o .env, rode:"
fi
echo "   docker compose up -d --build"
echo "   docker compose logs -f bot"
