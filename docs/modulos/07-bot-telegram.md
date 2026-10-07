# Bot do Telegram e Privacidade

**Responsável:** Rodrigo Atila Tavares Oliveira · **Código:** `src/fato_unb/bots/`

## Como usar

| Ação | O que acontece |
|---|---|
| `/checar <afirmação>` | Checa o texto escrito depois do comando |
| Responder uma mensagem com `/checar` | Checa o texto da mensagem respondida |
| `@bot <afirmação>` | Checa o texto da menção |
| `/privacidade` | Explica como os dados são tratados |
| `/help`, `/start` | Ajuda e boas-vindas |

A resposta mostra o veredito com emoji, a justificativa, a confiança e a lista de fontes com link.

## Fluxo

1. Chega um comando ou uma menção (o bot não recebe outras mensagens do grupo).
2. O texto passa por `scrub()`, que troca dados pessoais por rótulos.
3. O `FactChecker` faz a checagem, numa thread separada para o bot continuar atendendo.
4. O veredito é formatado em MarkdownV2 e enviado, sem pré-visualização de links.

Sem chave de API configurada, o bot responde `INCONCLUSIVO` explicando o problema.

## Privacidade

- **Só lê o que é endereçado a ele.** Com o *Privacy Mode* do Telegram ativo, o bot só recebe comandos, menções e respostas a ele ([Guiding Questions](../guiding_questions.md)). Conversas paralelas do grupo não chegam ao servidor.
- **Remoção de dados pessoais** (`privacy.py`), antes de qualquer análise:

| Dado | Vira |
|---|---|
| CPF (formatado ou 11 dígitos) | `[CPF]` |
| Telefone | `[TELEFONE]` |
| Número de 9 dígitos (matrícula) | `[MATRICULA]` |
| E-mail | `[EMAIL]` |

- **Logs também são limpos:** um filtro (`FiltroPII`) aplica o mesmo `scrub` em toda mensagem de log.
- O bot não guarda mensagens do grupo.

## Testes

`tests/test_bots.py` cobre a remoção de dados pessoais, a extração da afirmação, o reconhecimento da menção e a formatação da resposta.

## Pontos a revisar

- Qualquer número de 8 a 11 dígitos é tratado como dado pessoal, inclusive um valor ou uma data sem separador.
- O texto do `/help` diz que o bot não lê a mensagem respondida, mas o código usa essa mensagem quando ela está disponível.
