from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
# ai response -> str
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# RAG chain builder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# load models
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# load pdf
pdf_path = "Web_development_Book.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()
# text chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)

chunks = text_splitter.split_documents(pages)

# vector store
persist_directory = "./chroma_db"
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name="pdf_collection"
)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system","""you are a helpful assistant that answers questions based on the retrieved documents.
    
    RULES:
    - answer the question from the retrieved documents only, do not use any prior knowledge.
    - Be concise and to the point in your answer.
    - use bullet points if the answer contains multiple points.
     
     IMPORTANT: If you don't know the answer, say you don't know. Do not try to make up an answer based on the retrieved documents.

     CONTEXT:
     {context}
     """),
    ("human","{input}")
])

documents_chain = create_stuff_documents_chain(llm=llm, prompt=rag_prompt)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})
rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=documents_chain)

question = str(input("enter your question: "))
response = rag_chain.invoke({"input": question})
print(response['answer'])