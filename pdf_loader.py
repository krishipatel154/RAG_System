# read pdfs
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings
warnings.filterwarnings("ignore")

pdf_path = "Web_development_Book.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

print(f"number of pages: {len(pages)}")
print(f"metadata of first page: {pages[0].metadata}")

print("total characters in the document: ", sum([len(page.page_content) for page in pages]))
print("total tokens in the document: ", sum([len(page.page_content) for page in pages])//4)

# text chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200, 
    length_function=len, 
    is_separator_regex=False
)

print(f"number of chunks: {len(text_splitter.split_documents(pages))}")