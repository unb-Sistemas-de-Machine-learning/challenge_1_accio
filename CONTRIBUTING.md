### 🛠️ Configuração do Ambiente

Para criar o ambiente virtual e instalar todas as dependências do projeto:

```bash
uv sync
```

---

### 📖 Como Usar o `uv`

#### 1. Executar comandos e scripts (`uv run`)
O `uv run` executa qualquer comando ou script automaticamente dentro do ambiente virtual do projeto, sem precisar ativá-lo manualmente:

```bash
# Rodar o servidor local de documentação (MkDocs)
uv run mkdocs serve

# Executar um script Python
uv run python meu_script.py
```

#### 2. Adicionar novas dependências (`uv add`)
Para instalar e registrar um novo pacote no `pyproject.toml` e atualizar o `uv.lock`:

```bash
# Adicionar pacote padrão
uv add nome-do-pacote

# Adicionar versão específica
uv add "pandas>=2.0.0"

# Adicionar dependência de desenvolvimento
uv add --dev pytest
```

#### 3. Remover dependências (`uv remove`)
```bash
uv remove nome-do-pacote
```

#### 4. Uso compatível com pip (`uv pip`)
O `uv` também oferece uma interface rápida compatível com os comandos tradicionais do `pip`:

```bash
# Instalar pacote via uv pip
uv pip install nome-do-pacote

# Instalar a partir de um requirements.txt
uv pip install -r requirements.txt

# Listar pacotes instalados
uv pip list

# Congelar versões instaladas
uv pip freeze
```

#### 5. Sincronizar o ambiente (`uv sync`)
Caso novos pacotes tenham sido adicionados por outros membros da equipe (no `pyproject.toml` / `uv.lock`), execute:

```bash
uv sync
```

