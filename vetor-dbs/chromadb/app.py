from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAIOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA

# load documents from data directory
loader = DirectoryLoader("data/", glob="*.txt", loader_cls=TextLoader)
documents = loader.load()

# split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# persist the chunks
chunks.persist("data/chroma_db")

# create embeddings
embeddings = OpenAIEmbeddings()

# create a chroma vector store
vectorstore = Chroma(persist_directory="data/chroma_db", embedding_function=OpenAIEmbeddings(),documents=chunks)

# Persist the vector store
vectorstore.persist()
vectorstore = None

# load the vector store
vectorstore = Chroma(persist_directory="data/chroma_db", embedding_function=OpenAIEmbeddings())

# create a chain that uses the vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#search the vector store
query = "What is the capital of France?"
docs = retriever.get_relevant_documents(query)
print(docs)

qa_chain = RetrievalQA.from_chain_type(llm=OpenAIOpenAI(model="gpt-3.5-turbo"), chain_type="stuff", retriever=retriever)

result = qa_chain({"query": "What is the capital of France?"})
print(result)