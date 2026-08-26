# Guiding Questions

1. Quais as fontes, links e endpoints que serão usadas pelos agentes para consulta?
    - https://noticias.unb.br - UnB Notícias - HTML
    - https://www.unb.br/ - Universidade de Brasília - HTML
    - https://www.deg.unb.br/ - Decanato de Graduação - HTML
    - https://saa.unb.br/ - Secretaria de Administração Acadêmica - HTML e PDFs para informativos
    - https://dpg.unb.br - Decanato de Pós-graduação - HTML
2. Será utilizada arquitetura multi-agente, se sim quais os agentes?
    - Multi-agentes:
        - Supervisor, recebe o prompt e rapassa para o consultor
        - Consultor, Devolve a precisão utilizando a base de conhecimentos RAG
3. Qual o modelo de LLM será utilizado (acessível e barato)?
    - Ainda não foi decidido.
4. Quão viável é a implementação de um bot em grupos do Whatsapp em um grupo universitário?
    - Não é viável. A implementação de um bot em grupos de WhatsApp requer uma autorização da Meta. Qualquer outra forma de contornar isso é viola os termos de serviço. Será utilizado o Telegram como forma de implementação do bot.
5. Quais guardrails devem ser definidos para a privacidade de usuários?
    - Ingestão e escuta:
        - Nunca deve processar, analizar ou registrar mensagens que não sejam direcionadas a ele via menção
        - Sanitização prévia de entrada: Mascaram dados pessoais identificáveis comuns em chats
    - Armazenhamento e logs:
        - Não persistência de mensagens brutas
        - Anonimização de identificadores de rede

    - Interação com LLM e provedores de nuvem:
        - Adicionar instruções para que a IA nunca repita, exponha ou confirme dados pessoais sensíveis
    - Transparência e concentimento (LGPD):
        - Explicar quais dados são processados temporariamente e explicar como são descartados após a resposta

6. Quais as diretrizes das plataformas para a integração de bots em grupo?
    - Telegram (Telegram Bot API)
        - O Telegram é a plataforma mais amigável e permissiva para bots em ambientes comunitários, possuindo diretrizes claras de conformidade:
            - Group Privacy Mode (Modo de Privacidade de Grupo):
                - Por padrão, a Telegram Bot Platform Policy exige que bots adicionados a grupos tenham o Privacy Mode ativado.
                - O que implica: O bot não recebe todo o tráfego do chat. Ele só enxerga mensagens que comecem com / (comandos), respostas diretas a mensagens dele (replies) ou mensagens em que seu @username seja explicitamente mencionado.
                - Conformidade do FatoUnB: O bot opera em total conformidade mantendo esse modo ativo, garantindo que conversas paralelas dos estudantes nunca cheguem ao servidor.

        - Proibição de Spam e Flooding:
            - O Telegram impõe limites estritos de taxa (rate limits): no máximo 20 mensagens por minuto no mesmo grupo e 30 mensagens por segundo globalmente.
            - O bot deve ser puramente reativo (responde apenas sob demanda) e não pode disparar mensagens não solicitadas aos membros.

        - Transparência e Identificação:
            - Bots devem ser claramente identificáveis como automações (sufixo _bot obrigatório no username).
            - A descrição inicial e o comando /help devem deixar claro o propósito do bot e suas limitações de uso.

        - WhatsApp (Meta for Developers / WhatsApp Business API)
            - A Meta possui diretrizes significativamente mais rigorosas para automação em grupos:
            - Política de Uso Comercial e Proibição de Bots Não Oficiais:
                - O uso de bibliotecas de engenharia reversa de Web WhatsApp (ex: Baileys, WPPConnect) viola os Termos de Serviço do WhatsApp, sujeitando os números conectados a banimento imediato e irreversível.
            - Limitações da Cloud API Oficial em Grupos:
                - A API oficial do WhatsApp (WhatsApp Cloud API) foi desenhada historicamente para suporte 1:1 (Customer Service).
                - O suporte a grupos na API oficial é restrito e exige verificação da empresa/instituição no Meta Business Manager, além de seguir a WhatsApp Business Messaging Policy.
            - Janela de Atendimento de 24 Horas:
            - Mensagens de formato livre só podem ser enviadas dentro de uma janela de 24 horas a partir da última interação do usuário.

            - Modelo de Interação Permitido:
                - É estritamente proibido o envio de mensagens em massa (broadcasts não solicitados) dentro de grupos. O bot deve aguardar que o usuário inicie o contato ou mencione o serviço.

7. Como ele buscará os dados? consulta web direta ou em base de conhecimentos (RAG)?
    - RAG com webscraping, feito a cada 1 hora.