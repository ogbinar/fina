from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import yaml
import os

# Load the API key from the YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


# Set the API key in the environment variables
os.environ["OPENAI_API_KEY"] = config['openai_api_key']

# Load Data
loader = UnstructuredFileLoader("FIRE.txt")
raw_documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(raw_documents)


# Load Data to vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)


# Save vectorstore 
vectorstore.save_local("vectorstore.index")