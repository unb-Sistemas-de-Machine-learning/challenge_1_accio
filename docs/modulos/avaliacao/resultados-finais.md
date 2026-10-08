# Resultados Finais

## Metas do projeto

| Objetivo | Meta | Situação |
|---|---|---|
| Acerto do veredito | Acima de 50% | ✅ **90,1%** (73 de 81 casos) |
| Retenção de usuários | ≥ 50% voltam 3 vezes ou mais | ⏳ Bot pronto, uso real ainda não medido |

## O que está pronto

- ✅ Coleta (crawler HTML/PDF, RSS, agendador de 1 hora)
- ✅ Staging em PostgreSQL e banco vetorial Qdrant com upsert sem duplicar
- ✅ Busca híbrida com filtros por fonte, semestre e data
- ✅ Veredito por LLM com guardrails
- ✅ Bot do Telegram com remoção de dados pessoais
- ✅ Avaliação da busca e do veredito
- 🟡 Reranker: implementado, desligado e não medido
- 🟡 RAGAS: 15 dos 20 exemplos do testset foram avaliados; resultados ainda preliminares
- ⏳ Comparação de LLM local contra nuvem

## Resultados RAGAS

Foram avaliadas **15 das 20 afirmações** do testset. A configuração da busca usada foi
`paraphrase-multilingual-MiniLM-L12-v2`, sem RRF, e o modelo do verificador foi
`gemini-3.1-flash-lite`.

O bot acertou **11 dos 15 vereditos (73,3%)**. As médias e medianas das quatro
métricas RAGAS nesses exemplos são:

| Métrica RAGAS | Média | Mediana |
|---|---:|---:|
| Fidelidade ao contexto | 0,148888889 | 0 |
| Cobertura do contexto | 0,464285714 | 0,5 |
| Precisão do contexto | 0,605555556 | 0,75 |
| Correção da resposta | 0,470416544 | 0,403400374 |

O e5 com RRF não foi usado porque faltava memória RAM enquanto os modelos locais de
avaliação estavam em execução. Foram avaliados apenas 15 exemplos devido à lentidão
para gerar o testset e para avaliar cada pergunta. Portanto, estes resultados são
preliminares e não devem ser tratados como uma medição conclusiva da qualidade geral
do bot ou como comparação direta entre configurações.

### Análise dos erros de veredito

Nos 15 testes, os **quatro vereditos incorretos foram falsos negativos**: nos testes
2, 3, 5 e 9, a resposta esperada era `CONFIRMADO_OFICIALMENTE`, mas o bot respondeu
`BOATO_SEM_REGISTRO`. Ou seja, afirmações verdadeiras no testset foram rejeitadas.

Uma causa provável é a recuperação não ter trazido evidência suficiente para o
verificador. Nesta execução foi usado `paraphrase-multilingual-MiniLM-L12-v2` sem
RRF; essa configuração pode ter deixado passar os trechos relevantes, levando o
bot a não encontrar apoio para as afirmações. Os resultados, por si só, não provam
que essa foi a causa dos quatro erros: seria necessário reavaliar esses mesmos casos
com e5 e RRF e comparar os contextos recuperados e os vereditos.

Consulte a tabela por teste em [Avaliação do RAG](avaliacao-rag.md).

!!! info "Planilha com os resultados"
    Acesse o [CSV da avaliação](https://docs.google.com/spreadsheets/d/1o5Rf38Njlrxv3jkOVhR_rFxg3QDUjkVoReAsCGxzYGk/edit?usp=sharing). (2 páginas)

## Próximos passos

Os pontos abertos estão em [Checagem com LLM](../06-checagem-llm.md). Os principais:

1. Extrair a data real de publicação (causa da única confirmação indevida).
2. Medir o reranker.
3. Reavaliar os quatro falsos negativos com e5 e RRF, comparando os contextos recuperados.
4. Avaliar os cinco exemplos restantes do testset RAGAS e revisar as respostas; mesmo completo, o conjunto sintético ainda exige revisão antes de generalizar os resultados.
5. Manter o agendador rodando, pois o índice parou em 16/09/2026.
6. Comparar LLMs (Gemini, Claude Haiku, modelo local).
