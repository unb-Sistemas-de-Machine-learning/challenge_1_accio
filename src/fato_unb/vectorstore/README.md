# 🗄️ Módulo de Banco Vetorial (`vectorstore`)

**Responsável Principal:** Angelo Araujo Cordova (Engenheiro de Vector Store & Storage)  
**Épico Vinculado:** `[STORAGE]`

---

## 🎯 Objetivo
Gerenciar a persistência de embeddings vetoriais e metadados contextuais no Qdrant, garantindo operações atômicas de `upsert` e consultas com filtros de metadados sem *downtime*.

---

## 📂 Estrutura de Arquivos
* `client.py`: Singleton/Factory de conexão com a instância do Qdrant (local via Docker ou nuvem).
* `collections.py`: Inicialização e migração de coleções, definição de métricas de distância (`Cosine`) e dimensões do vetor.
* `operations.py`: Métodos de escrita idempotente (`upsert_documents`) e busca vetorial com filtros pré-busca (`source`, `semester_ref`, datas).

---

## 🚀 Subindo a Infraestrutura Local
```bash
# Subir o Qdrant via Docker Compose
docker compose up -d qdrant

    Dashboard web do Qdrant: http://localhost:6333/dashboard
```
---
# 🧪 Como Testar Este Módulo

```bash
# Rodar testes de integração com o Qdrant
uv run pytest tests/test_vectorstore.py -v
```