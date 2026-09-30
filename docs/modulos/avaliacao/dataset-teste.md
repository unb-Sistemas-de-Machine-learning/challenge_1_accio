# Dataset de Teste (recuperação)

Arquivo: `src/fato_unb/evaluation/retrieval_dataset.jsonl` (95 casos, um JSON por linha).

## Objetivo

Comparar configurações de chunking, embedding e reranking pela **qualidade da recuperação**:
dado o texto que o usuário mandaria ao bot, a evidência correta aparece entre os primeiros chunks?

## Composição

| Tipo | Casos | O que representa | Entra no recall/MRR |
|---|---|---|---|
| `verdadeira` | 43 | Afirmação fiel a uma notícia | sim |
| `falsa` | 27 | Afirmação com um dado alterado (número, data, órgão, entidade) | sim (a evidência é a notícia que a contradiz) |
| `desatualizada` | 4 | Fato que já mudou ou prazo encerrado | sim |
| `pergunta` | 14 | Pergunta em linguagem natural, inclusive coloquial e com sigla | sim |
| `sem_registro` | 7 | Boato sem fonte no corpus (alguns com confundidor lexical) | não (só registra o score do top-1) |

### Campo `desafio` (o que torna o caso difícil)

| Desafio | Casos | Exemplo |
|---|---|---|
| `direto` | 56 | Alegação que reusa os termos da notícia |
| `confundidor` | 14 | Existe outro documento parecido (TF/DCS do 1º vs 2º semestre; 60mais 216/218/224 vagas; CAUC vs NWPU) |
| `sigla` | 6 | CAD, Consuni, Semuni, CAUC; documentos de uma linha só |
| `numerico` | 5 | Número trocado ou de outro contexto (R$ 1,2 mi vs R$ 12 mi; 2.813 vs 2.117) |
| `parafrase` | 4 | Sem termos em comum ("Ministério da Justiça" no lugar de "Senajus") |
| `coloquial` | 4 | "quantas vagas tem pro PAS 3 pra 2027?" |
| `temporal` | 2 | Prazo já encerrado; regra que só vale a partir de 2027 |
| `multi_doc` | 2 | Evidência espalhada em dois documentos |
| `entidade` | 2 | Nome/unidade trocada |

## Campos

`id`, `tipo`, `alegacao`, `expected_urls`, `veredito_esperado`, `categoria`, `dificuldade`, `desafio`, `evidencia`, `evidence_spans`
(trecho da fonte que sustenta o rótulo, para facilitar a revisão humana).

## Como o acerto é medido

- O rótulo aponta **URLs**, resolvidas para o **conteúdo** do documento. Assim, cópias do mesmo texto com URLs
  diferentes (`.../#main`, `.../#`) contam como o mesmo documento.
- `recall@k`: alguma das `k` primeiras chunks recuperadas pertence a um documento esperado.
- `MRR`: média de 1/posição do primeiro chunk correto.
- Corpus: `dados.txt`, indexado numa coleção isolada em memória (não toca a coleção de produção).

## Métricas por chunk e por contexto

Além do recall por documento, cada caso tem `evidence_spans`: trechos **exatos** da fonte que respondem ao caso
(validados por teste contra o corpus). Isso permite medir:

- **C@k (chunk):** algum dos `k` primeiros chunks recuperados contém a evidência.
- **X@k (contexto):** algum dos `k` primeiros contextos contém a evidência, onde o contexto é o `parent_text`
  do chunk (o que o LLM receberia) ou o próprio chunk quando não há `parent_text`.
- **ctx(p):** palavras de contexto somadas nos 3 primeiros resultados (custo em tokens; sem deduplicar sobreposição).

Um span partido entre dois chunks conta como erro do chunker, que é justamente o que se quer medir.
Casos `multi_doc` aceitam qualquer um dos spans (evidência em algum dos documentos).

## Como rodar

```bash
uv run python scripts/avaliar.py --config baseline --falhas
uv run python scripts/avaliar.py --config baseline --raw   # sem deduplicar o corpus
```

## Limitações conhecidas

- Ainda é pequeno: com ~88 casos avaliáveis, diferenças de 2 a 3 pontos entre configurações são ruído.
  Só conte ganhos grandes e consistentes (ex.: em R@1 ou nos desafios `confundidor` e `temporal`).
- O corpus tem só 99 documentos únicos, então recall@5 tende ao teto. Compare principalmente **R@1 e MRR**.
- Rótulos escritos por IA a partir do corpus: cada caso traz o trecho da fonte em `evidencia`, mas ainda
  precisam de revisão humana. Os vereditos de `falsa` misturam `BOATO_SEM_REGISTRO` e
  `DESATUALIZADO_OU_FORA_DE_CONTEXTO` (número que pertence a outro semestre); vale confirmar a convenção.
- Casos `desatualizada` e `temporal` assumem "hoje" como fim de setembro de 2026.
- O corpus não contém nenhum documento da ADUnB.
