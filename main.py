# PYTHON
import os
import json
from pathlib import Path

# TELEGRAM
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# LANGCHAIN
from langchain.document_loaders import JSONLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap

# TRULENS
from trulens_eval.feedback.provider import OpenAI
from trulens_eval import Select
from trulens_eval import TruChain, Feedback, Huggingface, Tru
from trulens_eval.schema import FeedbackResult
from trulens_eval.feedback import Groundedness
from trulens_eval import LiteLLM
from trulens_eval import Langchain
from langchain.llms import VertexAI

import numpy as np
import requests

from google.cloud import aiplatform
from google.oauth2 import service_account

with open(os.path.join(Path(__file__).resolve().parent, 'keys.json')) as secrets_file:
    secrets = json.load(secrets_file)


def get_secret(key, json=secrets):
    try:
        return json[key]
    except KeyError:
        print("Error")


TELEGRAM_KEY = get_secret('TELEGRAM_KEY')
BOT_USERNAME = get_secret('BOT_USERNAME')
GEMINI_KEY = get_secret('GEMINI_KEY')



# credentials = service_account.Credentials.from_service_account_file("../credentials.json")

# aiplatform.init(
#     project="app",
#     location="us-central1",
#     credentials=credentials
# )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello')


def handle_response(text):
    print(f'User message: {text}')

    docs = vector_index.get_relevant_documents(text)

    # pprint(docs)

    prompt_template = "act as an employee of Shoppu store that is serving a client and answer all the client " \
                      "questions about the shop inventory, product descriptions, prices, existences, " \
                      "and availability, also answer questions related to products category names. Always answer in " \
                      "the same language the question is in.  If there is any answer that is not inside the context " \
                      "you must say: 'answer not available in context'. " \
                      "\n Contexts: {context} \n Question: {question} \Answer: "

    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = RunnableMap({
        "context": lambda x: vector_index.get_relevant_documents(x["question"]),
        "question": lambda x: x["question"],
    }) | prompt | model | output_parser

    response = chain.invoke({"question": text})

    # tru_recorder = TruChain(chain,
    #                         app_id='ShoppuChat Bot',
    #                         feedbacks=[f_qa_relevance, f_context_relevance, f_groundedness])
    #
    # with tru_recorder as recording:
    #     response = chain.invoke({"question": text})
    #     print("RECORDING")
    #     print(recording.get())

    print(f'Bot Message: {response}')
    return response


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type = update.message.chat.type
    text = update.message.text

    # if message_type == 'group':
    #     if BOT_USERNAME in text:
    #         new_text = text.replace(BOT_USERNAME, '').strip()
    #         response = handle_response(new_text)
    #     else:
    #         return
    #
    # else:

    response = handle_response(text)
    print("ANSWERING")
    await update.message.reply_text(response)
    print(f'{response}')


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pass


def shoppu_login():
    auth_data = {'username': get_secret('SHOPPU_USER'),
                 'password': get_secret('SHOPPU_PASSWORD')}

    return requests.post(get_secret('SHOPPU_URL'), json=auth_data).json()


def get_shoppu_products(token):
    response = requests.get(get_secret('SHOPPU_PRODUCTS'), headers={"Content-Type": "application/json",
                                                                    "Authorization": f'Token {token}'})

    with open("products.json", "w") as outfile:
        json.dump(response.json(), outfile)

    return response.json()


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["category"] = record.get("category").get("name")
    metadata["name"] = record.get("name")
    metadata["price"] = record.get("price") + '$'
    metadata["existence"] = record.get("existence")
    metadata["available"] = record.get("available")
    metadata["image"] = record.get("image")
    metadata["date_created"] = record.get("date_created")
    metadata["date_modified"] = record.get("date_modified")

    return metadata


if __name__ == '__main__':
    print('STARTING BOT...')

    login_json = shoppu_login()
    shoppu_token = login_json["token"]

    print('GETTING PRODUCTS FROM API...')
    products_json = get_shoppu_products(shoppu_token)

    print('SETTING MODEL...')
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.7,
                                   google_api_key=GEMINI_KEY)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                              google_api_key=GEMINI_KEY)

    print('LOADING JSON...')
    loader = JSONLoader(
        file_path='products.json',
        jq_schema='.[]',
        content_key="description",
        text_content=False,
        metadata_func=metadata_func)

    data = loader.load()

    print('CREATING EMBEDDINGS...')
    vector_index = Chroma.from_documents(documents=data, embedding=embeddings).as_retriever()
    output_parser = StrOutputParser()

    chat_history = []

    # print("SETTING TRULENS")
    # tru = Tru()
    # tru.reset_database()
    #
    # gemini_provider = LiteLLM(model_engine="gemini-pro")
    # grounded = Groundedness(groundedness_provider=gemini_provider)
    #
    # f_groundedness = (
    #     Feedback(grounded.groundedness_measure_with_cot_reasons)
    #     .on(Select.RecordCalls.first.invoke.rets.context)
    #     .on_output()
    #     .aggregate(grounded.grounded_statements_aggregator)
    # )
    #
    # f_qa_relevance = Feedback(gemini_provider.relevance).on_input_output()
    #
    # f_context_relevance = (
    #     Feedback(gemini_provider.qs_relevance)
    #     .on(Select.RecordCalls.first.invoke.args.input)
    #     .on(Select.RecordCalls.first.invoke.rets.context)
    #     .aggregate(np.mean)
    # )


    print('STARTING TELEGRAM APP...')
    app = Application.builder().token(TELEGRAM_KEY).build()
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.add_error_handler(error)

    print('BOT READY!')
    print('POLLING STARTED...')
    app.run_polling(poll_interval=3)
