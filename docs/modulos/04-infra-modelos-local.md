# Infra e Modelos

## Onde cada parte roda

| Parte | Tecnologia | Onde |
|---|---|---|
| Banco vetorial | Qdrant `v1.19.0` | Docker, porta 6333 |
| Staging | PostgreSQL 16 (SQLite como alternativa local) | Docker, porta 5432 |
| Embeddings, BM25 e reranker | fastembed (ONNX) | **Local**, CPU |
| Veredito | Gemini (ou Anthropic) | **Nuvem** |
| Agendador | APScheduler | Processo local |
| Bot | python-telegram-bot | Processo local |
| Documentação | MkDocs Material | GitHub Actions → GitHub Pages |

O projeto usa `uv` e Python ≥ 3.14.

## Modelos

| Modelo | Para quê | Situação |
|---|---|---|
| `intfloat/multilingual-e5-large` | Embedding denso | **Padrão** |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embedding denso | Usado nas primeiras fases |
| `Qdrant/bm25` | Vetor esparso | **Padrão** |
| `jinaai/jina-reranker-v2-base-multilingual` | Reranker | Implementado, desligado |
| `gemini-3.1-flash-lite` | Veredito | **Padrão**, avaliado |
| `claude-haiku-4-5` | Veredito | Implementado, ainda não executado de verdade |

O `gemini-2.5-flash-lite` não foi usado porque responde 404 para contas novas (comentário em `.env.example`).

## Avaliação com testset sintético

`src/fato_unb/evaluation/generateTestset.py` usa **Ollama** com modelos locais
(`qwen2.5:7b`, `qwen2.5:3b` e `bge-m3`) e RAGAS para gerar perguntas e afirmações a partir
de `dados.txt`. O avaliador está integrado em `src/fato_unb/evaluation/evaluator.py`.
O arquivo de testset atualmente salvo tem 20 exemplos, embora o gerador solicite 30 em
`testset_size`; as métricas RAGAS disponíveis e os resultados preliminares estão descritos em
[Métricas e Benchmark](05-metricas-benchmark.md).

## Por que essa divisão

- **Embeddings locais:** rodam a cada indexação e a cada consulta, e assim não têm custo por chamada nem dependem de cota.
- **LLM na nuvem:** o veredito precisa seguir instruções e devolver JSON válido. Os custos e a cota estão em [Checagem com LLM](06-checagem-llm.md).

Comparação detalhada em [Nuvem vs. Local](avaliacao/comparacao-nuvem-local.md).
