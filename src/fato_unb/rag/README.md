# 🧠 Módulo do Pipeline RAG & Checagem (`rag`)

**Responsável Principal:** Yan Santos Rodrigues (Engenheiro de IA & Pipeline RAG)  
**Épico Vinculado:** `[RAG-ENGINE]`

---

## 🎯 Objetivo
Segmentar documentos, gerar representações vetoriais, executar busca híbrida (Dense + BM25) e orquestrar a geração de vereditos estritos de checagem com LLMs sem alucinações.

---

## 📂 Estrutura de Arquivos
* `chunker.py`: Divisão em frases, chunks de ~120 palavras com 1 frase de overlap e `parent_text` (sentence-window).
* `embeddings.py`: Wrapper unificado para provedores de embedding (Google Gemini API / HuggingFace Local).
* `hybrid_search.py`: Algoritmo de fusão por ranking recíproco (RRF) combinando vetores e correspondência léxica exata.
* `prompts.py`: Templates estritos de fact-checking com regras de citação de fontes primárias.
* `engine.py`: Orquestrador central: `verificar_afirmacao(texto) -> VereditoJSON`.

---

## 📋 Categorias de Veredito
1. `CONFIRMADO_OFICIALMENTE`
2. `BOATO_SEM_REGISTRO`
3. `DESATUALIZADO_OU_FORA_DE_CONTEXTO`
4. `INCONCLUSIVO`

---

## 🧪 Como Testar Este Módulo
```bash
# Rodar testes de chunking e templates de prompt
uv run pytest tests/test_rag.py -v
```
---