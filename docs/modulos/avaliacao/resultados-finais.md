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
- 🟡 Testset RAGAS: 10 perguntas geradas, métricas não calculadas
- ⏳ Comparação de LLM local contra nuvem

## Próximos passos

Os pontos abertos estão em [Checagem com LLM](../06-checagem-llm.md). Os principais:

1. Extrair a data real de publicação (causa da única confirmação indevida).
2. Medir o reranker.
3. Manter o agendador rodando, pois o índice parou em 16/09/2026.
4. Comparar LLMs (Gemini, Claude Haiku, modelo local).
