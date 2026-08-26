# Objetivo de Negócio

Reduzir incerteza em informações compartilhadas em grupos da UnB

### Métrica: 

Taxa de retenção dos usuários no bot (vezes que o usuário voltou), mais que 50%  dos usuários voltaram ao bot mais de 3 vezes

# Objetivo de ML

Verificar se uma mensagem possui veracidade com base nos dados de informativos de canais oficiais da Universidade de Brasília

### Métrica: 

Taxa de confidence do modelo acima de 50% significa que deu certo.

# Escopo

> Nosso sistema trata mensagens de texto relacionadas a informações da UnB de grupos no wpp/telegram em português 
> 
> Não trata imagens, mensagens de audio e chamadas

# Guiding Questions

* 1-) Quais as fontes, links e endpoints que serão usadas pelos agentes para consulta 
> - https://noticias.unb.br - UnB Notícias - HTML
> - https://www.unb.br/ - Universidade de Brasília - HTML
> - https://www.deg.unb.br/ - Decanato de Graduação - HTML
> - https://saa.unb.br/ - Secretaria de Administração Acadêmica - HTML e PDFs para informativos
> - https://dpg.unb.br - Decanato de Pós-graduação - HTML
* 2-) Será utilizada arquitetura multi-agente, se sim quais os agentes? 
> Multi-agentes:
>
> - Supervisor, recebe o prompt e rapassa para o consultor
>
> - Consultor, Devolve a precisão utilizando a base de conhecimentos RAG

* 3-) Qual o modelo de LLM será utilizado (acessível e barato)?
> a
* 4-) Quão viável é a implementação de um bot em grupos do Whatsapp em um grupo universitário?
> a
* 5-) Quais guardrails devem ser definidos para a privacidade de usuários?
> Ingestão e escuta:
>
> - Nunca deve processar, analizar ou registrar mensagens que não sejam direcionadas a ele via menção
>
> - Sanitização prévia de entrada: Mascaram dados pessoais identificáveis comuns em chats
>
> ---
>
> Armazenhamento e logs:
>
> - Não persistência de mensagens brutas
>
> - Anonimização de identificadores de rede
>
> ---
>
> Interação com LLM e provedores de nuvem:
>
> - Adicionar instruções para que a IA nunca repita, exponha ou confirme dados pessoais sensíveis
>
> ---
>
> Transparência e concentimento (LGPD):
>
> - Explicar quais dados são processados temporariamente e explicar como são descartados após a resposta
* 6-) Quais as diretrizes das plataformas para a integração de bots em grupo?
> a
* 7-) Como ele buscará os dados? consulta WEB ou em base de conhecimentos (RAG)
> RAG com WEB Scraping, feito a cada 1 hora.
