# Avaliação do RAG

Só entram aqui números que já estão registrados em arquivos do repositório. As métricas estão explicadas em [Métricas e Benchmark](../05-metricas-benchmark.md).

## Busca

| Comparação | Resultado | Onde está registrado |
|---|---|---|
| Chunker novo (120 palavras, por frases) contra o antigo (400 palavras) | Praticamente igual: MRR 0,836 contra 0,853; R@1 0,739 contra 0,761 | `docs/modulos/03-pipeline-rag.md` (versão original) |
| e5-large contra MiniLM | R@1 de 0,761 para 0,943 (95 casos), com ~3,7× mais tempo por consulta e 2,2 GB de memória | Comentário em `src/fato_unb/rag/embeddings.py` |

Por isso o e5-large é o modelo padrão.

!!! note "Ainda não medido"
    O reranker (configs `e5-c120-rr`, `e5-c400-rr`) e o `jina-embeddings-v3` (configs `jina3-*`) existem em `CONFIGS`, mas o repositório não tem resultados deles.

## Veredito

Gemini 3.1 Flash-Lite, índice e5, 81 casos rotulados (resultados completos em [Checagem com LLM](../06-checagem-llm.md#resultado-gemini-31-flash-lite-indice-e5-81-casos-rotulados)).

| Métrica | Valor |
|---|---|
| **Acerto geral** | **90,1%** (73 de 81) |
| Acerto entre os respondidos | 93,6% (3 `INCONCLUSIVO`) |
| `falsa` e `sem_registro` | 100% |
| `verdadeira` | 83,7% |
| `desatualizada` | 75,0% (3 de 4) |
| Confirmações indevidas | 1 (caso c043) |
| Custo | US$ 0,075 nos 81 casos |

### Onde estão os 8 erros

- Em **4**, a busca trouxe o documento certo mas o trecho errado: **erro de busca, não do LLM**.
- Nos outros 4, a evidência estava no contexto: o modelo foi estrito demais, as fontes tinham dados conflitantes ou o rótulo é discutível.
- O erro grave (c043) confirmou um aviso de covid de 2021 como atual, porque a página estava com a **data da coleta** como data de publicação.

## Testset RAGAS

O `ragas_testset_local.csv` (branch `feat/evaluation`) tem 10 perguntas, todas de um mesmo tipo (abstrata de múltiplas etapas), e algumas se repetem. As métricas do RAGAS ainda não foram calculadas.
