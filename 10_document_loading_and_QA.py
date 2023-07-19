# Document loading and QA retrieval
# Using the data on your own data
import os
from dotenv import load_dotenv

# load in the env variables
load_dotenv()

# get the env variable relevant and needed
os.getenv('OPENAI_API_KEY')

# these are all the tools we have access to for our agents
from langchain import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# loading in the file or text document
loader = TextLoader('./readme.txt')
document = loader.load()

# split the document into chunks
# this means that its small enough so it loads in
# splits by double and one line and space
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=5)
texts = text_splitter.split_documents(document)

# embeddings
# A vast library of text embeddings, vector base data around multi factors to find how they relate
# i.e cows, chickens will be closer being animals while cars, bike could be vehicles
# taking words and sentacnes, mapping them to how they relate and new things will be closer together
embeddings = OpenAIEmbeddings()

# vector store
# Chroma, opensource, lightwight embedding database
# Stores to a local chroma database that we can query
store = Chroma.from_documents(texts, embeddings, collection_name="langchain-setup-and-read-me")

# query
# setup open ai then use the LLM and use Retrieval QA to perform data retrieval on the stored data
llm =OpenAI(temperature=0)
chain = RetrievalQA.from_chain_type(llm, retriever=store.as_retriever())

# input to check to query the data
print("What do you want to know about the document?")
user_input = input()
print(user_input)
# chain retrieves and finds relevence from the vector store to find a response
print(chain.run(user_input))