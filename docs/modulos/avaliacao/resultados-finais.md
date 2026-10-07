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
- 🟡 RAGAS: testset atual com 20 exemplos; somente 2 foram avaliados até agora, portanto as médias são preliminares
- ⏳ Comparação de LLM local contra nuvem

## Resultado preliminar de RAGAS

O arquivo atual `resultados_teste.csv` contém duas respostas avaliadas do testset de 20 exemplos.
As médias registradas são: faithfulness **0,000**, context recall **0,550**, context precision
**0,708**, answer correctness **0,476** e acurácia do veredito **0,500** (1/2).
Com apenas duas observações, esses números não são resultados finais nem permitem concluir sobre a
qualidade geral do bot. Veja o detalhamento em [Avaliação do RAG](avaliacao-rag.md).

## Próximos passos

Os pontos abertos estão em [Checagem com LLM](../06-checagem-llm.md). Os principais:

1. Extrair a data real de publicação (causa da única confirmação indevida).
2. Medir o reranker.
3. Completar e revisar a avaliação RAGAS nas 20 afirmações do testset atualmente salvo; as médias só devem ser interpretadas após avaliar o conjunto completo.
4. Manter o agendador rodando, pois o índice parou em 16/09/2026.
5. Comparar LLMs (Gemini, Claude Haiku, modelo local).
