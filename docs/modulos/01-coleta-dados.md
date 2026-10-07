# Coleta de Dados

**Responsável:** Pedro Henrique Inacio dos Santos · **Código:** `src/fato_unb/ingestion/`

O bot só consegue confirmar ou desmentir o que foi coletado. Se um comunicado não entrou na base, o veredito tende a ser "sem registro".

## Fontes

O crawler (`crawler.py`) parte destas páginas e só segue links dos mesmos domínios:

| Domínio | O que é |
|---|---|
| `noticias.unb.br` | UnB Notícias (HTML e RSS) |
| `dpg.unb.br` | Decanato de Pós-Graduação |
| `saa.unb.br` | Secretaria de Administração Acadêmica (calendários, HTML e PDF) |
| `deg.unb.br` | Decanato de Graduação |
| `adunb.org` | ADUnB (sindical, não é fonte oficial da UnB) |

## Como coleta

- **Crawler:** visita as páginas em lotes de 5, com pausa de 3 s entre os lotes, e ignora URLs já salvas. Descarta documentos publicados antes de **01/01/2026** e páginas com 50 palavras ou menos.
- **HTML (`html.py`):** extrai o texto principal com `trafilatura`. A data vem da meta tag `article:published_time`, da tag `<time>` ou da própria URL.
- **PDF (`pdf.py`):** extrai o texto de cada página com PyMuPDF. PDFs sem data são descartados.
- **RSS (`rss.py`):** o feed só traz um resumo, então, para cada notícia nova, o texto completo é baixado da página.

## Documento coletado (`RawDocument`)

| Campo | Observação |
|---|---|
| `doc_id` | SHA-256 da URL (a mesma URL sempre gera o mesmo ID) |
| `title`, `content`, `url`, `source` | Título, texto, endereço e origem |
| `source_type` | `html_page`, `pdf_document` ou `rss_news` |
| `published_at` | Data de publicação |
| `semester_ref` | Calculado da data: meses 1 a 7 → `AAAA.1`, meses 8 a 12 → `AAAA.2` |

## Agendamento

`scheduler.py` roda a cada 1 hora: crawler → RSS → grava no staging → indexa no Qdrant. Cada documento novo também é anexado a `dados.txt` (141 documentos hoje), usado como corpus nos testes e na avaliação.

## Limitação conhecida

Se a página não informa a data, o código usa a **data da coleta** como data de publicação (`html.py`). Isso causou o único erro grave da avaliação (ver [Checagem com LLM](06-checagem-llm.md)).
