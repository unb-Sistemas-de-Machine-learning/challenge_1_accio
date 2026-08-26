# 📊 Módulo de Avaliação & Benchmarks (`evaluation`)

**Responsável Principal:** Matheus Pinheiro (Engenheiro de QA, Avaliação & Documentação)  
**Épico Vinculado:** `[QA-EVALUATION]`

---

## 🎯 Objetivo
Validar quantitativamente a acurácia e a taxa de confiança da IA contra a meta de projeto (>50%), além de conduzir benchmarks comparativos de latência, consumo e custo entre modelos em nuvem e locais.

---

## 📂 Estrutura de Arquivos
* `dataset.json`: Dataset rotulado (*Ground Truth*) contendo boatos históricos da UnB, fatos oficiais, fontes esperadas e vereditos humanos.
* `evaluator.py`: Pipeline de avaliação automatizada medindo métricas de RAG (*Faithfulness*, *Context Recall*, *Answer Relevance*).
* `benchmark.py`: Coleta de métricas de engenharia (Time to First Token - TTFT, Latência E2E, consumo de RAM/VRAM e custos por query).

---

## 🧪 Como Executar a Avaliação
```bash
# Rodar avaliação do dataset contra o pipeline RAG
uv run python -m fato_unb.evaluation.evaluator

# Executar o benchmark comparativo Nuvem vs Local
uv run python -m fato_unb.evaluation.benchmark
```
---