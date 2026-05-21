from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

warnings.filterwarnings("ignore")

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

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

sample_embedding = embeddings.embed_query("What is web development?")
# print(sample_embedding)

persist_directory = "./chroma_db"
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name="pdf_collection"
)

query = "What is backend development?"
results = vector_store.similarity_search(query, k=3)
print(results)

# retriever = vector_store.as_retriever()