# Métricas e Benchmark

**Código:** `src/fato_unb/evaluation/` e `scripts/`

Um veredito errado pode ter duas causas: a **busca** não trouxe a evidência certa, ou o **LLM** interpretou mal. Por isso a avaliação tem duas partes.

## 1. Busca (`scripts/avaliar.py`)

Usa o dataset de 95 casos ([Dataset de Teste](avaliacao/dataset-teste.md)) sobre o corpus `dados.txt`, numa coleção temporária em memória que não toca a de produção.

| Métrica | Pergunta |
|---|---|
| **R@k** | O documento certo apareceu entre os `k` primeiros resultados? |
| **MRR** | Em média, em que posição ele apareceu? (1 = primeiro) |
| **C@k** | O **trecho** com a evidência apareceu? |
| **X@k** | A evidência estava no contexto que o LLM recebe? |

Cada configuração (modelo, tamanho do chunk, reranker...) está em `CONFIGS`, no arquivo `retrieval.py`:

```bash
uv run python scripts/avaliar.py --config baseline chunk120-idf e5-c120 --falhas
```

## 2. Veredito (`scripts/avaliar_vereditos.py`)

Roda o fluxo completo (busca → LLM → guardrails) nos 81 casos que têm veredito esperado e compara.

| Métrica | Por que importa |
|---|---|
| Acerto geral | A meta do projeto é acima de 50% |
| `INCONCLUSIVO` | O bot não pode se abster sempre |
| **Confirmações indevidas** | O erro mais grave: confirmar algo falso ou desatualizado |
| Acerto por tipo e matriz de confusão | Mostram onde o sistema erra |
| Causa do erro | A frase com a evidência chegou ao LLM? Separa erro de busca de erro do modelo |
| Custo e tempo | Tokens e latência por checagem |

A data de "hoje" é fixa (`--hoje`, padrão 2026-09-30), porque casos de prazo dependem dela. Os resultados são gravados em JSONL, e uma nova execução continua de onde parou.

```bash
uv run python scripts/avaliar_vereditos.py --falhas
```

Os resultados estão em [Avaliação do RAG](avaliacao/avaliacao-rag.md).

## 3. Testset sintético (RAGAS)

O `generateTestset.py` (branch `feat/evaluation`, Matheus) gera perguntas de teste a partir de `dados.txt` com modelos locais e salva em `ragas_testset_local.csv`. O `README.md` do módulo prevê as métricas *Faithfulness*, *Context Recall* e *Answer Relevance*, que ainda não foram calculadas.
