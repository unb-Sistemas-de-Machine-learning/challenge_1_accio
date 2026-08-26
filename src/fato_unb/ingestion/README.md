# 📥 Módulo de Ingestão e Webscraping (`ingestion`)

**Responsável Principal:** Pedro Henrique Inacio dos Santos (Engenheiro de Dados & Scraping)  
**Épico Vinculado:** `[DATA-INGESTION]`

---

## 🎯 Objetivo
Coletar, limpar e padronizar comunicados, notícias institucionais, notas sindicais e editais em PDF de fontes oficiais da UnB (SECOM, DEG, SAA, ADUnB, SINTFUB, DCE).

---

## 📂 Estrutura de Arquivos
* `models.py`: Schemas canônicos com Pydantic (`RawDocument`, `SourceType`) e cálculo de `doc_id` determinístico (hash SHA-256 da URL).
* `rss.py`: Coletor de alta frequência para feeds RSS/Atom (`feedparser`).
* `html.py`: Extrator de texto limpo via `trafilatura` com fallback para `BeautifulSoup4`.
* `pdf.py`: Extrator de texto de editais e calendários usando `PyMuPDF` (`fitz`).
* `scheduler.py`: Agendador horário de tarefas em segundo plano com `APScheduler`.

---

## 🔌 Contrato de Saída
Todas as funções de extração devem retornar instâncias ou listas de `RawDocument`:

```python
RawDocument(
    title="Calendário Acadêmico 2026/1 Aprovado",
    content="O Conselho de Ensino, Pesquisa e Extensão...",
    url="[https://noticias.unb.br/](https://noticias.unb.br/)...",
    source="UnB Notícias",
    source_type=SourceType.RSS_NEWS,
    published_at=datetime(2026, 3, 1, 10, 0),
    semester_ref="2026/1"
)
```
---
# 🧪 Como Testar Este Módulo

```bash
# Rodar testes unitários de ingestão
uv run pytest tests/test_ingestion.py -v
```