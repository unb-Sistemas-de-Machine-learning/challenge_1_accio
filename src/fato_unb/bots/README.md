# 🤖 Módulo de Bots & Mensageria (`bots`)

**Responsável Principal:** Rodrigo Atila Tavares Oliveira (Engenheiro de Infra, Modelos & Bots)  
**Épico Vinculado:** `[BOTS-INFRA]`

---

## 🎯 Objetivo
Disponibilizar as interfaces de chat para a comunidade universitária (Telegram e WhatsApp) sob conformidade com a LGPD e o princípio de *Zero-Overhearing*.

---

## 📂 Estrutura de Arquivos
* `privacy.py`: Sanitizador de PII que remove CPFs, matrículas UnB (`\d{9}`), telefones e e-mails antes de qualquer log ou envio para a LLM.
* `telegram_bot.py`: Implementação do bot do Telegram com handlers para `/checar`, `/privacidade` e menções diretas em grupos e filtro de menção (`@bot`) para envio de mensagens via *quoted message*.
---

## 🔒 Regras de Privacidade
* **Zero-Overhearing:** Mensagens onde o bot não é mencionado são descartadas imediatamente da memória.
* **Sem Persistência de Mensagens Brutas:** O banco de dados nunca armazena o corpo do chat ou números de telefone dos usuários.

---

## 🧪 Como Testar Este Módulo
```bash
# Rodar testes de sanitização de PII e handlers
uv run pytest tests/test_bots.py -v
```
---
