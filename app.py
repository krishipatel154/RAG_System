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

# streamlit
import streamlit as st
import tempfile
import os

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# page config
st.set_page_config(page_title="RAG with Groq LLM", page_icon=":books:", layout="wide")

# initialize session state - memory for chat history and vector store
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

@st.cache_resource
def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = get_llm()
embeddings = get_embeddings()

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

def process_pdf(pdf_file):
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name

    with st.spinner("Splitting into chunks..."):
        # load pdf
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()

        # text chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )

        chunks = text_splitter.split_documents(pages)

    with st.spinner("Creating vector store..."):
        persist_dir = os.path.join(tempfile.gettempdir(), "chroma_db")

        # vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name="streamlit_pdf"
        )
    
    os.unlink(tmp_file_path)
    return vector_store
    
st.title("RAG with Groq LLM :books:")
st.caption("Upload a PDF and ask questions about its content!")

with st.sidebar:
    st.header("Upload PDF")
    pdf_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader", help="Upload a PDF document to enable question answering based on its content.")

    if pdf_file is not None and not st.session_state.pdf_processed:
        vector_store = process_pdf(pdf_file)

        st.session_state.vector_store = vector_store
        st.session_state.pdf_processed = True
        st.session_state.chat_history = []  # reset chat history when new PDF is uploaded

        st.success("PDF processed successfully! You can now ask questions about its content.")

    if st.session_state.pdf_processed:
        st.info("PDF is already processed. To upload a new PDF, please refresh the page.")

    if st.session_state.pdf_processed:
        if st.button("Upload New PDF"):
            st.session_state.pdf_processed = False
            st.session_state.vector_store = None
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.markdown("**Instructions:**\n1. Upload a PDF document.\n2. Wait for it to be processed.\n3. Ask questions about the content in the input box below.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask a question about the PDF content...", disabled=not st.session_state.pdf_processed)

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)

        st.session_state.chat_history.append({"role": "user", "content": user_question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                document_chain = create_stuff_documents_chain(llm=llm, prompt=rag_prompt)
                retriever = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})
                rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)
                response = rag_chain.invoke({"input": user_question})
                answer = response['answer']

            st.markdown(answer)
            
            with st.expander("View Retrieved Documents"):
                for i, doc in enumerate(response["context"], 1):
                    page_num = doc.metadata.get("page_number", "N/A")
                    st.markdown(f"**Document {i+1}:**")
                    st.caption(doc.page_content[:500] + "...")
                    st.divider()

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# empty state
if not st.session_state.pdf_processed:
    st.info("Please upload a PDF document to start asking questions about its content.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1 Upload")
        st.markdown("Upload a PDF document using the sidebar uploader.")
    with col2:
        st.markdown("### 2 Ask Questions")
        st.markdown("Once the PDF is processed, ask questions about its content in the input box.")

    with col3:
        st.markdown("### 3 Get Answers")
        st.markdown("Receive concise answers based on the retrieved content from the PDF.")
    