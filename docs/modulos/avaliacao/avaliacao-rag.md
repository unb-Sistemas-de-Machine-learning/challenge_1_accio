# Avaliação do RAG

Esta página separa resultados de recuperação, avaliação do veredito e a execução preliminar de RAGAS. As métricas e os comandos estão em [Métricas e Benchmark](../05-metricas-benchmark.md).

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

O `ragas_testset_local.csv` atualmente registrado tem **20 exemplos**: 10 rotulados `VERDADEIRO` e 10 `FALSO`; são 10 perguntas específicas de etapa única e 10 de múltiplas etapas. O script gerador solicita 30 exemplos, então a quantidade atual do CSV não deve ser tomada como o tamanho garantido de uma nova geração.

Resultados de **15 testes**:

| Teste | Fidelidade ao contexto | Cobertura do contexto | Precisão do contexto | Correção da resposta |
|---:|---:|---:|---:|---:|
| 1 | 0 | 0,5 | 0,416666667 | 0,38771298 |
| 2 | 0 | 0 | 1 | 0,375905679 |
| 3 | 0 | 0,5 | 0,333333333 | 0,486824344 |
| 4 | 0 | 0,5 | 1 | 0,547839233 |
| 5 | 0,333333333 | 0,5 | 0 | 0,609355509 |
| 6 | 0 | 1 | 0,25 | 0,347145339 |
| 7 | 0 | 0 | 1 | 0,766936671 |
| 8 | 0,75 | 1 | 0,333333333 | 0,732893995 |
| 9 | 0 | 0,333333333 | 1 | 0,381738537 |
| 10 | 0 | 0,5 | 1 | 0,403400374 |
| 11 | 0,25 | 0 | 0,75 | 0,401825812 |
| 12 | 0,4 | 0,666666667 | 1 | 0,536376433 |
| 13 | 0 | — | 0 | 0,525915744 |
| 14 | 0,5 | 0,666666667 | 0 | 0,165215931 |
| 15 | 0 | 0,333333333 | 1 | 0,387161577 |
| **Média** | **0,148888889** | **0,464285714** | **0,605555556** | **0,470416544** |
| **Mediana** | **0** | **0,5** | **0,75** | **0,403400374** |

`—` indica que a cobertura do contexto não tem valor registrado no teste 13. Os valores agregados de cobertura consideram os resultados disponíveis. A [planilha detalhada com os resultados](https://docs.google.com/spreadsheets/d/1o5Rf38Njlrxv3jkOVhR_rFxg3QDUjkVoReAsCGxzYGk/edit?usp=sharing) também pode ser consultada.

Para entender o significado das métricas, consulte [Métricas e Benchmark](../05-metricas-benchmark.md). Os passos para gerar o testset e executar a avaliação pelo terminal estão em [Como Executar](../../execucao.md#8-criar-o-testset-e-avaliar-com-ragas).
