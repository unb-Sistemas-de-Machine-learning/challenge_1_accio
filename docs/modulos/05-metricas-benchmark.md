# Métricas e Benchmark

**Código:** `src/fato_unb/evaluation/` e `scripts/`

Um veredito errado pode ter duas causas: a **busca** não trouxe a evidência certa, ou o **LLM** interpretou mal. O repositório tem duas avaliações supervisionadas do fluxo e uma trilha complementar com RAGAS.

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

O dataset de recuperação tem 95 casos; 88 têm documento esperado e entram em recall/MRR. Os 7 casos
`sem_registro` ficam fora dessas métricas, pois não têm documento esperado.

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

## 3. Qualidade da resposta com RAGAS (`src/fato_unb/evaluation/evaluator.py`)

O gerador `src/fato_unb/evaluation/generateTestset.py` usa os documentos de `dados.txt` e modelos locais via Ollama para criar perguntas sintéticas e afirmações avaliáveis. O CSV atualmente presente no repositório tem 20 exemplos: 10 rotulados `VERDADEIRO` e 10 `FALSO`, com 10 perguntas específicas de etapa única e 10 de múltiplas etapas. O código do gerador solicita `testset_size=30`; portanto, a quantidade produzida em uma nova execução pode diferir do arquivo atualmente salvo.

No modo `--bot`, o avaliador envia a **afirmação** (não a pergunta sintética original) ao verificador, registra resposta, fontes e contextos recuperados, e compara o veredito com o rótulo. As métricas RAGAS configuradas são:

| Métrica | O que avalia |
|---|---|
| `faithfulness` | Se as afirmações da resposta são sustentadas pelos contextos recuperados |
| `context_recall` | Se os contextos recuperados cobrem as informações da referência |
| `context_precision` | Se os contextos relevantes aparecem bem posicionados |
| `answer_correctness` | Similaridade factual entre a resposta estruturada e a referência |

### Fórmulas

As fórmulas abaixo descrevem a implementação RAGAS usada pelo projeto. O RAGAS usa o LLM juiz
para decompor respostas/referências em afirmações ou classificar a relevância; portanto, os
indicadores `0` e `1` são julgamentos do modelo, não correspondências de texto calculadas
literalmente. Cada métrica é calculada por exemplo.

**Faithfulness — fidelidade ao contexto**

Seja \(S_a\) o conjunto de afirmações extraídas da resposta e \(s_i=1\) quando o juiz determina que
a afirmação \(i\) é sustentada pelos contextos recuperados (caso contrário, \(s_i=0\)):

\[
\text{Faithfulness} =
\frac{\sum_{i=1}^{|S_a|} s_i}{|S_a|}
\]

**Context Recall — cobertura do contexto**

Seja \(S_r\) o conjunto de afirmações extraídas da referência e \(r_i=1\) quando a afirmação \(i\)
pode ser atribuída aos contextos recuperados:

\[
\text{Context Recall} =
\frac{\sum_{i=1}^{|S_r|} r_i}{|S_r|}
\]

**Context Precision — precisão e ordenação dos contextos**

Para cada contexto recuperado na posição \(k\), \(v_k=1\) se o juiz o considera relevante para a
afirmação de entrada e a referência, e \(v_k=0\) caso contrário. A precisão acumulada no prefixo até
\(k\) é \(P@k=(\sum_{i=1}^{k}v_i)/k\). A métrica é a precisão média dos contextos relevantes:

\[
\text{Context Precision} =
\frac{\sum_{k=1}^{n} (P@k \cdot v_k)}{\sum_{k=1}^{n}v_k}
\]

Assim, contextos relevantes no início da lista contribuem mais para a nota. A ordem avaliada é a
ordem em que os contextos são entregues ao avaliador.

**Answer Correctness — correção da resposta**

O juiz compara afirmações extraídas da resposta e da referência e classifica correspondências
como verdadeiros positivos (\(TP\)), afirmações incorretas/adicionais como falsos positivos
(\(FP\)) e afirmações da referência omitidas como falsos negativos (\(FN\)). Com \(\beta=1\), o
componente factual é o F1:

\[
F1 = \frac{2TP}{2TP + FP + FN}
\]

O segundo componente, \(Sim\), é a similaridade do cosseno entre os embeddings da resposta e da
referência. A implementação usa os pesos padrão de 0,75 para factualidade e 0,25 para similaridade:

\[
\text{Answer Correctness} = 0.75 \cdot F1 + 0.25 \cdot Sim
\]

Em caso sem afirmações extraídas da resposta e da referência, o RAGAS trata o componente factual
como 1; casos sem afirmações extraídas da resposta para `faithfulness` podem resultar em valor
indefinido (`NaN`).

O avaliador também calcula `acuracia_do_veredito`, que é uma comparação direta com o rótulo e não uma métrica RAGAS. O juiz usa `qwen2.5:3b` e os embeddings do avaliador usam `bge-m3`, ambos via Ollama. `answer_relevancy` não faz parte da configuração atual.

O CSV e o resumo atualmente registrados (`resultados_teste.csv` e `resultados_teste_resumo.json`) contêm somente **2 de 20** exemplos avaliados. As médias dessa amostra são: faithfulness 0,000; context recall 0,550; context precision 0,708; answer correctness 0,476; acurácia do veredito 0,500 (1/2). São resultados preliminares de uma amostra muito pequena, não uma medição final nem evidência suficiente para comparar modelos. O modo de busca mede recuperação contra evidência rotulada; o modo de veredito mede o fluxo real com LLM; RAGAS pontua a resposta e os contextos em um testset sintético. Não se deve misturar os denominadores ou interpretar essas métricas como equivalentes.

Consulte [Avaliação do RAG](avaliacao/avaliacao-rag.md), [Dataset de Teste](avaliacao/dataset-teste.md) e [Resultados Finais](avaliacao/resultados-finais.md) para os números e limitações.
