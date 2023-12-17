import os
import json
from pathlib import Path

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

import google.generativeai as genai


with open(os.path.join(Path(__file__).resolve().parent, 'keys.json')) as secrets_file:
    secrets = json.load(secrets_file)


def get_secret(key, json=secrets):
    try:
        return secrets[key]
    except KeyError:
        print("Error")


TELEGRAM_KEY = get_secret('TELEGRAM_KEY')
BOT_USERNAME = get_secret('BOT_USERNAME')
GEMINI_KEY = get_secret('GEMINI_KEY')


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello')


def handle_response(text):
    print(f'User message: {text}')
    response = model.generate_content(text)
    print(f'response: {response.text}')
    return response.text


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type = update.message.chat.type
    text = update.message.text

    if message_type == 'group':
        if BOT_USERNAME in text:
            new_text = text.replace(BOT_USERNAME, '').strip()
            response = handle_response(new_text)
        else:
            return

    else:
        response = handle_response(text)

    await update.message.reply_text(response)


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pass


if __name__ == '__main__':
    print('STARTING BOT')

    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-pro')

    app = Application.builder().token(TELEGRAM_KEY).build()

    app.add_handler(CommandHandler('start', start_command))

    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    app.add_error_handler(error)

    print('STARTING POLLING')
    app.run_polling(poll_interval=3)