# Checagem com LLM e guardrails

Código em `src/fato_unb/llm/`; uso em `scripts/checar.py`.

## Fluxo

```
alegação
  → guardrails de entrada     valida tamanho, sinaliza prompt injection
  → Retriever (e5-large)      até 4 evidências com contexto (parent_text)
  → LLM                       prompt com evidências numeradas [1]..[4], datas e fontes
  → guardrails de saída       valida JSON, fontes e citações; rebaixa o veredito se não se sustentar
  → VereditoJSON              veredito + justificativa + fontes + confiança
```

O verificador responde **somente com o que está indexado**: não consulta a internet na hora da pergunta e o
prompt proíbe o modelo de usar conhecimento próprio. Fato que não está no índice tende a virar "sem registro".

**Provedor e modelo:** `LLM_PROVIDER` (`gemini` | `anthropic`) e `LLM_MODEL`. Padrão: `gemini` /
`gemini-3.1-flash-lite` ($0,25 / $1,50 por 1M de tokens; ~US$ 0,001 por checagem). O `gemini-2.5-flash-lite`
responde 404 para contas novas. Chaves no `.env` (ignorado pelo git): `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`.

## Guardrails implementados (primeira versão)

| Momento | Código | O que faz |
|---|---|---|
| Entrada | `alegacao_invalida` | Rejeita vazia, com menos de 8 ou mais de 500 caracteres; não chama o LLM |
| Entrada | `injecao_suspeita` | Só sinaliza frases como "ignore as instruções"; o texto já vai ao modelo como dado, com as tags de delimitação neutralizadas |
| Entrada | `sem_evidencias` | INCONCLUSIVO sem chamar o LLM |
| Entrada | `evidencia_fraca` | **Desligado** até calibrar o limiar do reranker |
| Saída | `fonte_inexistente` | O LLM citou uma evidência que não existe: removida |
| Saída | `citacao_nao_verificada` | A citação precisa aparecer literalmente na evidência citada (mínimo 12 caracteres) |
| Saída | `sem_fonte_ou_citacao_valida` | CONFIRMADO ou DESATUALIZADO sem fonte e citação verificáveis vira INCONCLUSIVO |
| Saída | `fonte_nao_oficial` | CONFIRMADO só com fontes fora de `*.unb.br` (ex.: ADUnB) vira INCONCLUSIVO |
| Saída | `justificativa_vazia` / `justificativa_truncada` | Justificativa vazia rebaixa; acima de 600 caracteres é truncada |
| Saída | `confianca_limitada` | Teto de 0,95; veredito rebaixado limita a 0,5 |
| Robustez | `json_reparado` / `json_invalido` | Uma nova tentativa se o JSON vier inválido; se falhar de novo, INCONCLUSIVO |
| Robustez | `llm_repetido` / `llm_indisponivel` | Até 3 tentativas (espera de 1 s e 3 s) só para 503, 429 e timeout; se persistir, INCONCLUSIVO |

Todo guardrail acionado fica registrado em `Checagem.guardrails`; um veredito rebaixado explica o motivo na
justificativa. `--bruto` mostra o prompt e a resposta crua do modelo antes dos guardrails.

## Primeiro teste real (Gemini 3.1 Flash-Lite + índice e5, 6 casos)

Os 6 vereditos saíram como esperado (boato, sem registro, 2 desatualizados, 1 confirmado, 1 injeção ignorada).
Com o modelo simulado há 33 testes automáticos. O adaptador da **Anthropic ainda não foi executado de verdade**
(não há chave).

## Pontos para revisão dos guardrails

**Decisões de desenho**
1. **Falta um veredito "falso/contrariado".** `BOATO_SEM_REGISTRO` cobre tanto "sem evidência" quanto "a fonte
   oficial diz o contrário", o que comunica mal ao usuário. Avaliar um 5º valor (ex.: `FALSO_CONTRARIADO_OFICIALMENTE`).
   `VereditoType` é contrato compartilhado com o bot e a documentação, então a mudança exige alinhamento.
   A diferença hoje: "já foi verdade, não é mais" = DESATUALIZADO; "nunca foi verdade" = BOATO.
2. **O veredito depende da data do sistema.** "Inscrições do PAS 3 abertas em 30/09" vira DESATUALIZADO; em 15/09
   seria CONFIRMADO. Garantir data correta no servidor e decidir se a data vem de fora (teste, reprocessamento).
3. **Escopo:** não há guardrail para alegações fora do tema UnB (ex.: assunto qualquer). Definir a resposta.
4. **Regra de fonte oficial** é `*.unb.br`. Decidir o papel da ADUnB (sindical): hoje nunca confirma sozinha.
   O nome da fonte também é inconsistente no índice (`UnB Notícias` e `noticias.unb.br`).

**Qualidade da resposta**
5. **A confiança do modelo não informa nada:** veio ≥ 0,95 em quase todos os casos. Substituir por um cálculo
   nosso (número de fontes, citações verificadas, nota do reranker).
6. **A justificativa pode perder o dado que contradiz:** em "IA oferta 200 vagas" o modelo disse que a evidência
   "não menciona" o número, em vez de dizer que o curso oferece 60.
7. **Verificação de citação é por texto exato:** pode rejeitar paráfrase legítima, e o contexto enviado é truncado
   em 1.800 caracteres por evidência. Medir a taxa de falsos rebaixamentos.
8. **`evidencia_fraca` está desligado.** Sem o reranker medido não há limiar, e o RRF não serve (só reflete
   posição). "Sem registro" hoje depende de o documento certo entrar no top-4.

**Dados e operação**
9. **O índice está parado em 16/09/2026 e o agendador não está rodando.** Decidir se haverá busca ao vivo como
   reserva quando a base não tiver nada relevante.
10. **Cobertura:** ADUnB e DEG não têm nenhum documento no índice, embora o crawler tenha as URLs iniciais.
11. **Latência irregular (2,5 s a 35 s)** por alta demanda do Gemini. Falta um tempo-limite total e uma resposta
    de espera para o bot; avaliar modelo ou provedor de reserva.
12. **Injeção:** a detecção é por padrões e só sinaliza. O texto das páginas coletadas também é entrada não
    confiável (as tags são neutralizadas, o conteúdo semântico não).
13. **Privacidade e retenção:** `Checagem.prompt_usuario` guarda a alegação e as evidências. Definir o que é logado.
    Conferir os termos de uso de dados do free tier do Gemini (não verifiquei).

**Medição**
14. **Não existe avaliador de vereditos.** O dataset de 95 casos já traz `veredito_esperado`; falta rodá-lo
    contra o LLM para medir o acerto (meta do projeto: > 50%) e comparar modelos (Gemini 3.1 Flash-Lite,
    Claude Haiku 4.5 etc.). Os rótulos de `falsa` misturam BOATO e DESATUALIZADO (ver item 1).
