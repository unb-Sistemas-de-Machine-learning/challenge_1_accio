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
* **Divisão Recursiva por Unidades Lógicas:** Prioriza a quebra por parágrafos duplos (`\n\n`), descendo para sentenças (`. `) e palavras (` `) apenas quando o bloco excede a janela alvo.
* **Overlap Calibrado:** Mantém uma janela deslizante (padrão: 40 a 50 palavras) entre blocos vizinhos, garantindo continuidade de leitura em regras compostas ou datas.
* **Injeção de Metadados no Conteúdo (`Context Injection`):** Cada pedaço recebe um cabeçalho fixo antes da vetorização:
  ```text
  [Documento: Circular Normativa DEG nº 02/2026]
  [Fonte: DEG | Ref: 2026/1]

  Art. 2º O período de ajuste extraordinário ocorrerá entre 10 e 15 de março...