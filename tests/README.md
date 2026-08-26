# 🧪 Suíte de Testes Automatizados (`tests/`)

Este diretório contém os testes unitários e de integração do projeto **FatoUnB**, executados via **Pytest**.

---

## 📂 Organização dos Testes
* `test_ingestion.py`: Testa parsers de RSS, extratores de HTML e leitores de PDF com dados mockados.
* `test_vectorstore.py`: Valida conexão com o Qdrant, inserção idempotente (`upsert`) e filtros de busca.
* `test_rag.py`: Testa o algoritmo de chunking, renderização de prompts e saídas de veredito da LLM.
* `test_bots.py`: Valida sanitização de dados pessoais (PII) e filtros de menção.
* `conftest.py`: Fixtures compartilhadas do Pytest (mocks de HTML, feeds RSS de exemplo e clientes de teste).

---

## 🚀 Como Executar os Testes com `uv`

```bash
# Executar todos os testes
uv run pytest

# Executar com saída detalhada e prints no terminal
uv run pytest -v -s

# Executar apenas testes de um módulo específico
uv run pytest tests/test_ingestion.py

# Gerar relatório de cobertura de código (Coverage)
uv run pytest --cov=src/fato_unb tests/
```