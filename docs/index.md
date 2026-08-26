# 🏛️ FatoUnB — Inteligência e Fact-Checking Universitário

> **Sistema automatizado de verificação factual, combate à desinformação e consulta a fontes oficiais da Universidade de Brasília (UnB).**

---

## 🎯 Objetivos do Projeto

<div class="grid cards" markdown>

-   :material-briefcase-check:{ .lg .middle } **Objetivo de Negócio**

    ---

    Reduzir a incerteza e mitigar a propagação de boatos em grupos e comunidades universitárias da UnB.

    **Métrica de Sucesso:**
    * Taxa de retenção $\ge 50\%$ (mais de 50% dos usuários que acionaram o bot retornam para consultá-lo pelo menos 3 vezes).

-   :material-robot:{ .lg .middle } **Objetivo de Machine Learning**

    ---

    Verificar a veracidade e consistência de afirmações textuais confrontando-as estritamente com os dados e informativos oficiais da UnB via RAG.

    **Métrica de Sucesso:**
    * Confiança (*confidence score*) do modelo de veredito $\ge 50\%$ nas análises sobre a base indexada.

</div>

---

## 🔍 Escopo do Sistema

* **Entradas Suportadas:** Mensagens de texto em português compartilhadas em grupos do Telegram/WhatsApp relacionadas a editais, calendários acadêmicos, avisos da Reitoria e serviços da UnB (RU, bibliotecas, transporte).
* **Fora de Escopo:** Processamento de imagens (OCR de prints), transcrição de mensagens de áudio, chamadas de voz e consultas sem relação com o ecossistema institucional da UnB.

---

## 🏗️ Arquitetura Geral do Sistema

```mermaid
flowchart LR
    A[Portais UnB / RSS] -->|Scraping a cada 1h| B(Ingestão & Sanitização PII)
    B -->|RawDocument| C(Chunking & FastEmbed)
    C -->|Vetores 384d| D[(Qdrant Vector DB)]
    E[Usuário / Grupo] -->|Menção ao Bot| F(Agente Supervisor)
    F -->|Repasse de Query| G(Agente Consultor RAG)
    D <-->|Recuperação Híbrida| G
    G -->|Veredito Estruturado| E
```

## 👥 Divisão de Módulos & Responsabilidades

| Módulo | Escopo & Descrição | Responsável |
|---|---|---|
| `ingestion` | Scrapers periódicos (1h), parsers HTML/PDF e sanitização de PII. | Pedro Henrique Inacio dos Santos |
| `storage` | Configuração do Qdrant, schemas de vetores e indexação híbrida. | Angelo Araujo Cordova |
| `rag` | Chunking semântico, embeddings locais (fastembed) e motor de vereditos. | Yan Santos Rodrigues |
| `interfaces` | Agente Supervisor, bots para Telegram/WhatsApp e API REST FastAPI. | Rodrigo Atila Tavares de Oliveira |
| `evaluation` | Métricas de retenção, avaliação de alucinação (RAGAS) e acurácia. | Matheus Pinheiro |

## 🚀 Como Executar Localmente

### 1. Clonar o Repositório e Sincronizar Dependências

```bash
git clone https://github.com/unb-Sistemas-de-Machine-learning/challenge_1_accio.git

cd challenge_1_accio

uv sync

uv pip install -e .
```

### 2. Rodar a Suíte de Testes
```bash
uv run pytest tests/ -v
```
### 3. Visualizar a Documentação
```bash
uv run mkdocs serve
```