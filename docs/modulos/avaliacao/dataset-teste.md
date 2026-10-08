# Dataset de Teste (recuperação)

Arquivo: `src/fato_unb/evaluation/retrieval_dataset.jsonl` (95 casos, um JSON por linha).

Este é o dataset de **recuperação**, usado por `scripts/avaliar.py`; não é o testset sintético
`ragas_testset_local.csv`, usado para avaliar respostas com RAGAS. As duas avaliações têm conjuntos,
objetivos e métricas diferentes.

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


## Testset sintético para RAGAS

O script `src/fato_unb/evaluation/generateTestset.py` gera outro conjunto de dados,
independente do dataset de recuperação acima. Ele lê `dados.txt` na raiz do projeto
como JSONL, transforma cada registro em um documento com título e conteúdo e remove
duplicatas pelo campo `doc_id`. Em seguida, usa as transformações padrão do RAGAS
para extrair informações e criar perguntas sintéticas com contextos de referência.

O gerador usa Ollama local: `qwen2.5:7b` para gerar o testset e as afirmações,
`qwen2.5:3b` para as transformações e `bge-m3` para embeddings. As instruções aos
modelos pedem textos em português brasileiro. O script solicita 30 exemplos, mas
o total efetivamente gerado pode ser menor ou variar conforme os documentos e o
processamento dos modelos.

Para cada exemplo gerado, o script distribui os rótulos `VERDADEIRO` e `FALSO` em
quantidades tão iguais quanto possível. A afirmação é criada a partir da pergunta
sintética e dos contextos de referência: deve ser sustentada diretamente por eles
quando verdadeira, ou contradizer um fato explícito quando falsa. O CSV também
registra metadados do sintetizador, personas quando disponíveis e as fontes
associadas aos contextos de referência.

### Limitações do testset sintético

- As instruções pedem português brasileiro e saídas estruturadas, mas o Qwen pode
  gerar conteúdo em outro idioma, fora do formato solicitado ou com erros factuais.
  A afirmação só é verificada pelo script quanto a ser texto não vazio; sua
  fidelidade aos contextos não é validada automaticamente. Revise as amostras e
  os rótulos antes de usar os resultados como evidência de qualidade.
- Se o RAGAS não conseguir interpretar a saída de uma extração de temas, o script
  registra um aviso e ignora aquele trecho; isso pode reduzir a quantidade de
  amostras geradas em relação às 30 solicitadas. Outros erros de geração podem
  interromper a execução.
- Metadados como persona, estilo e comprimento dependem do que foi produzido para
  cada amostra e podem ficar ausentes no CSV.
- A geração depende dos modelos e do corpus disponíveis localmente, pode ser
  demorada e não garante exatamente a mesma quantidade de exemplos em cada
  execução.

### Como gerar

Execute a partir da raiz do repositório, onde ficam `dados.txt` e o arquivo de
saída. Instale as dependências opcionais e obtenha os modelos necessários:

```bash
uv sync --extra evaluation --dev
ollama pull qwen2.5:7b
ollama pull qwen2.5:3b
ollama pull bge-m3
```

Confirme que o serviço Ollama está em execução e então rode:

```bash
uv run python -m fato_unb.evaluation.generateTestset
```

O script cria ou **sobrescreve** `ragas_testset_local.csv` na raiz. Esse CSV é a
entrada da avaliação RAGAS, não o dataset de recuperação
`src/fato_unb/evaluation/retrieval_dataset.jsonl` nem o CSV de resultados da
avaliação. Consulte [Métricas e Benchmark](../05-metricas-benchmark.md) e
[Como Executar](../../execucao.md#8-criar-o-testset-e-avaliar-com-ragas) para
entender a avaliação e executar o avaliador.
