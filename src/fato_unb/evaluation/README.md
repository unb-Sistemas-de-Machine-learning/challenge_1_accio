# 📊 Módulo de Avaliação & Benchmarks (`evaluation`)

**Responsável Principal:** Matheus Pinheiro (Engenheiro de QA, Avaliação & Documentação)  
**Épico Vinculado:** `[QA-EVALUATION]`

---

## 🎯 Objetivo
Validar quantitativamente a acurácia e a taxa de confiança da IA contra a meta de projeto (>50%), além de conduzir benchmarks comparativos de latência, consumo e custo entre modelos em nuvem e locais.

---

## 📂 Estrutura de Arquivos
* `dataset.json`: Dataset rotulado (*Ground Truth*) contendo boatos históricos da UnB, fatos oficiais, fontes esperadas e vereditos humanos.
* `evaluator.py`: Avalia respostas do bot e também permite executar uma avaliação local do testset.
* `benchmark.py`: Coleta de métricas de engenharia (Time to First Token - TTFT, Latência E2E, consumo de RAM/VRAM e custos por query).

---

## 🔎 Avaliação de Recuperação (retrieval)
* `dataset.py` / `retrieval_dataset.jsonl`: casos de teste (alegação → URL da evidência) e carregamento do corpus.
* `retrieval.py`: `recall@k` e `MRR` por configuração (`CONFIGS`), sobre uma coleção Qdrant isolada em memória.
* CLI: `uv run python scripts/avaliar.py --config baseline --falhas`. Detalhes em `docs/modulos/avaliacao/dataset-teste.md`.

---

## 🧪 Como Executar a Avaliação
```bash
# Instalar as dependências opcionais pesadas de avaliação
uv sync --extra evaluation

# Gerar os testsets Ragas e do bot a partir de dados.txt
uv run python -m fato_unb.evaluation.generateTestset

# Gerar respostas de teste com o Qwen e avaliar fidelidade/cobertura
uv run python -m fato_unb.evaluation.evaluator

# Avaliar a resposta atual do bot para a primeira afirmação
uv run python -m fato_unb.evaluation.evaluator --bot --limit 1

# Avaliar as respostas atuais do bot para todas as afirmações do CSV
uv run python -m fato_unb.evaluation.evaluator --bot

# Retomar a avaliação interrompida (use os mesmos modelos, modo e arquivos)
uv run python -m fato_unb.evaluation.evaluator --bot --resume

# Avaliar em etapas: as 5 primeiras e depois ampliar para as 10 primeiras
uv run python -m fato_unb.evaluation.evaluator --bot --limit 5 --output resultados_avaliacao.csv
uv run python -m fato_unb.evaluation.evaluator --bot --limit 10 --output resultados_avaliacao.csv --resume

# Testar somente a primeira pergunta no modo de geração local
uv run python -m fato_unb.evaluation.evaluator --limit 1

# Executar o benchmark comparativo Nuvem vs Local
uv run python -m fato_unb.evaluation.benchmark
```

O gerador salva o testset em `ragas_testset_local.csv` e acrescenta as colunas
`afirmacao` e `VEREDITO`. A afirmação é criada pelo Qwen com base na pergunta
gerada e nos contextos de referência. Para dividir os vereditos igualmente, o
gerador cria 20 amostras: 10 recebem `VEREDITO=VERDADEIRO`, com afirmação
sustentada pelos contextos, e 10 recebem `VEREDITO=FALSO`, com afirmação
diretamente contradita por eles.

O avaliador lê `ragas_testset_local.csv` na raiz do projeto. O modo padrão
continua gerando respostas locais para a coluna `pergunta`. Com `--bot`, usa a
coluna `afirmacao` como entrada (não a pergunta que originou o exemplo), chama
a mesma busca e montagem de veredito do Telegram e avalia a resposta estruturada
real, incluindo o veredito e as fontes citadas. Os contextos passados ao Ragas
são os trechos recuperados pela busca e usados para montar a resposta.

### Arquivos CSV e JSON

| Arquivo | Tipo | Conteúdo |
| --- | --- | --- |
| `ragas_testset_local.csv` | CSV de entrada | Testset gerado para avaliação. Contém perguntas/afirmações, vereditos esperados, contextos e respostas de referência. Não é o resultado da avaliação. |
| `resultados_avaliacao.csv` | CSV de resultados em lote | Uma linha por exemplo avaliado com sucesso: dados do testset, resposta do bot/modelo, fontes e contextos recuperados, veredito e métricas RAGAS. Linhas com erro são omitidas. Inclui `indice_linha_testset` para retomar sem duplicar avaliações. |
| `resultados_avaliacao_resumo.json` | JSON de resumo | Médias das métricas das linhas salvas no CSV de resultados. É atualizado ao final de cada execução concluída. |
| `resultados_avaliacao.csv.checkpoint.json` | JSON de checkpoint | Metadados para validar testset, modo, modelos e métricas ao usar `--resume`. Não contém respostas nem notas de avaliação. |
| `resultados_respostas_bot.csv` | CSV de avaliação individual | Registros anexados por `avaliar_resposta_do_bot` quando uma resposta do Telegram é avaliada individualmente contra uma afirmação do testset. |

Os nomes de saída em lote são definidos por `--output`: o caminho informado é o
CSV de resultados; o avaliador cria ao lado dele `<nome>_resumo.json` e
`<nome>.csv.checkpoint.json` (por exemplo, `--output minha_execucao.csv` gera
`minha_execucao_resumo.json` e `minha_execucao.csv.checkpoint.json`). Sem
`--output`, o CSV e o resumo padrão ficam no diretório
`src/fato_unb/evaluation/`. Na execução salva como `parte.csv`, esse arquivo é
o CSV de resultados descrito acima e seu resumo correspondente é
`parte_resumo.json`; `parte.csv.checkpoint.json` é apenas o checkpoint.

Antes de executar com `--bot`, inicie o Qdrant e confirme que a coleção
`fato_unb_noticias` está disponível e populada. Na raiz do projeto:

```bash
docker compose up -d qdrant
```

O serviço local usa por padrão `localhost:6333`; veja também as instruções do
módulo `vectorstore`. Sem conexão com o Qdrant, a busca de cada afirmação falha,
a linha é registrada no log e não entra no CSV. Depois de restabelecer o serviço,
use `--resume` com o mesmo arquivo de saída para tentar essas linhas novamente.

O modo `--bot` salva as respostas, fontes e contextos, além das métricas de
fidelidade ao contexto, cobertura e precisão dos contextos, correção da resposta
e acurácia do veredito, em
`resultados_avaliacao.csv`.
Atualmente, a implementação provisória produz `INCONCLUSIVO`; assim, a
acurácia do veredito mostra a diferença em relação ao rótulo do testset até que
o classificador esteja ativo. O testset atual só rotula `VERDADEIRO` e `FALSO`,
mapeados respectivamente para `CONFIRMADO_OFICIALMENTE` e
`BOATO_SEM_REGISTRO` para calcular essa acurácia. Os dois rótulos não avaliam
separadamente `DESATUALIZADO_OU_FORA_DE_CONTEXTO` e `INCONCLUSIVO`; quando o
testset passar a conter essas categorias, elas serão comparadas diretamente
com os quatro valores de `VereditoType`.

Para avaliar uma resposta individual do bot antes de enviá-la, chame
`avaliar_resposta_do_bot` passando a afirmação e os contextos recuperados:

```python
from fato_unb.evaluation.evaluator import avaliar_resposta_do_bot
from fato_unb.bots.telegram_bot import verificar_afirmacao_provisoria_com_contextos

veredito, contextos = verificar_afirmacao_provisoria_com_contextos(afirmacao)
metricas = await avaliar_resposta_do_bot(
    afirmacao=afirmacao,
    response_json=veredito.model_dump(mode="json"),
    retrieved_contexts=contextos,
)
texto = formatar_veredito(veredito)
await update.message.reply_text(texto)
```

Essa função só pontua correspondências únicas da coluna `afirmacao`, ignorando
maiúsculas, acentos e espaços extras; afirmações fora do testset retornam `None`
e geram um aviso. Ela inclui veredito e fontes na resposta avaliada e anexa
resultados a `resultados_respostas_bot.csv`.

Os dois modos de avaliação calculam ainda estas métricas RAGAS:

* `context_precision` (`precisao_do_contexto`) verifica se os contextos
  recuperados relevantes estão bem posicionados.
* `answer_correctness` (`correcao_da_resposta`) compara a resposta gerada com a
  referência factual do testset.

As notas são gravadas por linha no CSV e suas médias no JSON de resumo. As
prompts dessas métricas são configuradas para português.
`answer_correctness` compara a resposta estruturada inteira, incluindo
veredito e fontes, com uma referência que combina o rótulo esperado e a resposta
factual. Interprete essa nota como diagnóstica enquanto o bot ainda retorna
sempre `INCONCLUSIVO`.

No modo em lote, cada linha é avaliada isoladamente. Se a busca, geração ou
qualquer métrica falhar para uma linha, ela é registrada no log e omitida do
CSV; as linhas seguintes continuam sendo processadas. Se todas falharem, o CSV
terá apenas o cabeçalho e o resumo terá métricas `null`. As chamadas por linha
podem elevar o tempo total. Comece com `--limit 1`; evite executar a avaliação
ao mesmo tempo que o bot atende mensagens se os recursos forem limitados.

O CSV de saída é atualizado a cada linha concluída. Se precisar parar, use
`Ctrl+C`; linhas já gravadas são preservadas. Para continuar, repita o comando
com `--resume`, mantendo o mesmo testset, modo (`--bot` ou padrão), modelos e
caminho de saída. Um arquivo `<saida>.checkpoint.json` valida essa configuração;
se algo tiver mudado, a retomada é recusada para evitar misturar resultados
incompatíveis. `--limit` pode ser aumentado na retomada para ampliar o lote.
Linhas que falharam não são marcadas como concluídas e serão tentadas novamente.
O CSV inclui `indice_linha_testset`, usado para não recalcular linhas concluídas;
o JSON de resumo é atualizado quando a execução termina.

O Ragas usa o Qwen 2.5 3B como juiz e BGE-M3 para embeddings. As chamadas de
avaliação podem demorar; por isso o fluxo recomendado para o testset é executá-
las offline com `--bot`, em vez de atrasar cada resposta do Telegram.

O juiz do Ollama é configurado em modo JSON para ajudar o RAGAS a interpretar
as respostas estruturadas das métricas.
---