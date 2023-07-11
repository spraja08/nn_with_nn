import os
import sys

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

model_name = "gpt-3.5-turbo"
code_file_namne = "context/nn_with_nn_wrong.py"
context_file_name = "context/context.txt"
query = None

code = open(code_file_namne).read()

loader = TextLoader(context_file_name)
index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(model=model_name),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}))

chat_history = []
query_count = 0
while True:
    query = input("Prompt: ")
    if query_count == 0:
        query += ": \n" + code + "\n"
        print(query)
    if query in ['quit', 'q', 'exit']:
        sys.exit()
    result = chain({"question": query, "chat_history": chat_history})
    print(result['answer'])

    chat_history.append((query, result['answer']))
    query_count += 1