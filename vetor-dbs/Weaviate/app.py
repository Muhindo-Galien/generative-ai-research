import os
from langchain.vectorstores import Weaviate
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAIOpenAI
from langchain.prompts import PromptTemplate
from weaviate.client import WeaviateClient
from weaviate.auth import weaviate

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
os.environ["WEAVIATE_API_KEY"] = "YOUR_WEAVIATE_API_KEY"
os.environ["WEAVIATE_CLUSTER"] = "YOUR_WEAVIATE_CLUSTER"

# load documents from data directory, using PyPDFDirectoryLoader
loader = PyPDFDirectoryLoader("data/", glob="**/*.pdf")
data = loader.load()

# split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(data)

# create embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

# create a weaviate client with a startup period of 10 seconds 
weaviate_client = WeaviateClient(
    url=os.environ["WEAVIATE_CLUSTER"],
    additional_headers={
        "X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]
    },
    auth_client_secret=weaviate.AuthApiKey(
        username=os.environ["WEAVIATE_API_KEY"]
    ),
   startup_period=10
)

# check if the weaviate client is connected
print(weaviate_client.is_ready())

# create a weaviate vector store
vectorstore = Weaviate.from_documents(
    documents=chunks,
    embedding=embeddings,
    weaviate_client=weaviate_client,
    
)

# create a chain that uses the vector store
