import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
# O httpx loga a URL de cada requisição, e a URL contém o token do bot.
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

TEXTO_AJUDA = (
    "Olá, sou o FatoUnB, o bot que te ajuda a identificar fakenews na UnB!\n\n" #! essa mensagem tá horrível, dps vou mudar
    "Como usar em grupos:\n"
    "• Responda a mensagem suspeita com /checar\n"
    "• Ou me mencione: @{username} <afirmação>\n\n"
    "Só leio mensagens em que sou chamado. /privacidade explica o que faço com os dados."
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Olá! Eu sou o FatoUnB. Use /help para ver como me usar."
    )


async def ajuda(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(TEXTO_AJUDA.format(username=context.bot.username))


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN não foi declarado ou está expirado")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", ajuda))

    logger.info("Bot iniciado. Ctrl+C para parar.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
