# 🧠 Módulo de RAG & Checagem (`fato_unb.rag`)

**Responsável:** Yan Santos Rodrigues (Engenheiro de IA & Pipeline RAG)  
**Épico Vinculado:** `[RAG-ENGINE]`  

---

## 🎯 Objetivo do Módulo

O módulo `rag` é o núcleo de processamento semântico do **FatoUnB**. Ele transforma documentos textuais semiestruturados (notícias, circulares do DEG, resoluções e editais da SAA) em representações vetoriais de alta precisão, preservando metadados contextuais e preparando a base para buscas híbridas e vereditos sem alucinações.

---

## 🏗️ Componentes Implementados

### 1. Modelos de Dados (`models.py`)
Contratos estritos baseados em **Pydantic v2** para garantir validação em tempo de execução e serialização determinística.

* **`DocumentChunk`**: Representa a unidade atômica indexada no banco vetorial.
  * `chunk_id`: Hash determinístico SHA-256 (`doc_id + chunk_index`) truncado em 16 caracteres.
  * `doc_id`: ID do documento original de origem.
  * `content`: Texto do chunk prefixado com o cabeçalho de contexto institucional.
  * `raw_text`: Trecho original extraído sem formatações adicionais.
  * `chunk_index` / `total_chunks`: Controle sequencial da posição no documento pai.
  * Metadados herdados: `title`, `url`, `source`, `semester_ref`.

* **`VereditoJSON`**: Formato padronizado de saída da análise factual.
  * `veredito`: Classificação estrita (`CONFIRMADO_OFICIALMENTE`, `BOATO_SEM_REGISTRO`, `DESATUALIZADO_OU_FORA_DE_CONTEXTO`, `INCONCLUSIVO`).
  * `justificativa`: Texto direto fundamentado exclusivamente nas evidências recuperadas.
  * `fontes`: Lista de links oficiais (`FonteCitada`) que embasam o veredito.
  * `confianca`: Pontuação de 0.0 a 1.0 indicando o grau de correspondência semântica.

---

### 2. Chunking Semântico com Injeção de Contexto (`chunker.py`)
Fatiador de texto projetado para resolver o problema de perda de contexto e corte abrupto de termos em editais e normas da UnB.

#### Características do Algoritmo:
* **Unidade básica = frase:** o texto é dividido por linha (`\n`, pois o trafilatura nunca gera `\n\n`) e depois por frases, sem quebrar em abreviações ("Profa.", "Art.", "J. Silva"). Frases maiores que o chunk são cortadas por palavras.
* **Chunk de ~120 palavras** (`chunk_size`), com **1 frase de overlap** (`overlap_sentences`). Uma cauda com poucas palavras novas (< `min_chunk_words`) é fundida ao chunk anterior.
* **Sentence-window (`parent_text`):** cada chunk guarda o trecho ao redor (~300 palavras, `parent_size`). A busca usa o chunk pequeno; o LLM deve receber o `parent_text`.
* **Injeção de Metadados no Conteúdo (`Context Injection`):** Cada pedaço recebe um cabeçalho fixo antes da vetorização:
  ```text
  [Documento: Circular Normativa DEG nº 02/2026]
  [Fonte: DEG | Ref: 2026/1]

  Art. 2º O período de ajuste extraordinário ocorrerá entre 10 e 15 de março...

> **Avaliação (fase 1):** no dataset de 95 casos, o chunker novo ficou estatisticamente neutro em relação ao anterior
> (400 palavras): MRR 0,836 contra 0,853; R@1 0,739 contra 0,761 (1 caso = 1,1 ponto). O ganho esperado é qualitativo
> (estrutura de parágrafos, `parent_text`) e ainda não é medido pelo recall por documento. Variantes comparáveis com
> `uv run python scripts/avaliar.py --config baseline chunk80 chunk120 chunk200`.
