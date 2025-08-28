from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAIOpenAI
from langchain.prompts import PromptTemplate
import os
import pinecone

os.environ["PINECONE_API_KEY"] = "YOUR_PINECONE_API_KEY"
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
os.environ["PINECONE_ENVIRONMENT"] = "YOUR_PINECONE_ENVIRONMENT"

# load documents from data directory, using PyPDFDirectoryLoader
loader = PyPDFDirectoryLoader("data/")
documents = loader.load()

# split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# create embeddings
embeddings = OpenAIEmbeddings()

# intialize pinecone
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT"]
)

index_name = "test-index"

# create a pinecone vector store
vectorstore = Pinecone.from_documents([t.page_content for t in chunks], embedding=embeddings, index_name=index_name)

# load the vector store
vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings) 

# simmilary search
retriever = vectorstore.similarity_search("What is the capital of France?", k=3)
#retrieve the documents in vector format
print(retriever)


# create an LLM Model wrapper
llm = OpenAIOpenAI()

# create a chain that uses the vector store
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever.as_retriever())

# create a chain that uses the vector store
result = qa.run({"query": "What is the capital of France?"})
print(result)