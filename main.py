import os
import platform

import openai
import chromadb
import langchain
import docx # to read docx files
import warnings
warnings.filterwarnings("ignore")
#test

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models import ChatOpenAI

import urllib
from urllib import request

from pprint import pprint
from flask import Flask, render_template, request
# Create Flask app
app = Flask(__name__)


#
# from google-colab import drive
#
# drive.mount('/content/drive')

os.environ["OPENAI_API_KEY"] = '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'

collection_name = "aillm"
persist_directory = "C:/aipersist"

loader = DirectoryLoader("C:/aichat")
kb_data = loader.load()

class DirectoryLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        documents = []
        for filename in os.listdir(self.path):
            if filename.endswith('.txt'):
                with open(os.path.join(self.path, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                documents.append(text)
            elif filename.endswith('.docx'):
                doc = docx.Document(os.path.join(self.path, filename))
                text = '\n'.join([para.text for para in doc.paragraphs])
                documents.append(text)
            else:
                print(f"Invalid file {os.path.join(self.path, filename)}. The file type is not supported.")
        return '\n'.join(documents)

text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
kb_doc = text_splitter.split_documents(kb_data)

embeddings = OpenAIEmbeddings()
kb_db = Chroma.from_documents(kb_doc, embeddings, collection_name=collection_name, persist_directory=persist_directory)
kb_db.persist()

kb_qa = ChatVectorDBChain.from_llm(ChatOpenAI(temperature=0.0,
                                          model_name="gpt-3.5-turbo"),
                                   vectorstore=kb_db,
                                   top_k_docs_for_context=1,
                                   return_source_documents=True)

chat_history = []
number = ""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def answer():
    # Get user input
    query = request.form['query']
    #query = query + " " + "Do not give me any information about the procedure and service features that are not mentioned in the PROVIDED CONTEXT"
    # Query the chatbot model
    result = kb_qa({"question": query, "chat_history": chat_history})
    # Extract answer and source documents
    answer = result['answer']
    source_documents = result['source_documents']
    # Render answer template with results
    return render_template('answer.html', query=query, answer=answer, source_documents=source_documents)

if __name__ == '__main__':
    app.run(debug=True)
    # pprint(result[“source_document”])