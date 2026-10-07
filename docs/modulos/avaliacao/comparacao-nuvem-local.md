# Nuvem vs. Local

## O que usamos hoje

| Parte | Onde roda |
|---|---|
| Embeddings, BM25 e reranker | **Local** (CPU) |
| Veredito (LLM) | **Nuvem** (Gemini; Anthropic implementado) |
| Geração de testset | **Local** (Ollama) |

## Local: embeddings

- Sem custo por chamada, sem cota e sem enviar o corpus para fora.
- Custo: o e5-large usa cerca de 2,2 GB de memória e é ~3,7× mais lento que o MiniLM (comentário em `embeddings.py`).

## Nuvem: veredito

Dados medidos com o Gemini 3.1 Flash-Lite ([Checagem com LLM](../06-checagem-llm.md)):

- **Acerto:** 90,1% em 81 casos.
- **Custo:** cerca de US$ 0,001 por checagem.
- **Latência:** média de 9,3 s e p95 de 21,5 s, com muita variação por causa da demanda no provedor.
- **Cota:** o free tier permite 15 requisições por minuto por modelo. Na primeira avaliação, falhas de cota fizeram o acerto parecer 39,5%.
- A afirmação sanitizada e as evidências saem da máquina; convém conferir os termos de uso do provedor.

## O que falta

A comparação **LLM local contra LLM em nuvem** para o veredito ainda não foi feita: só o Gemini foi medido, e o adaptador da Anthropic nunca foi executado. Para fechar, seria preciso rodar `scripts/avaliar_vereditos.py` com cada modelo e comparar acerto, confirmações indevidas, custo e tempo.
