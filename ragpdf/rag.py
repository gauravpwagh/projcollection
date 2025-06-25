!pip install streamlit PyPDF2 langchain transformers sentence-transformers faiss-cpu
!pip install -U langchain-community



import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
import tempfile
import os

# Pre-download embedding model to cache for offline use
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")

# --- Load LLM with llama-cpp-python ---
@st.cache_resource
def load_llm():
    return LlamaCpp(
        model_path="mistral-7b-instruct.gguf",  # Path to your local GGUF model
        n_ctx=2048,
        temperature=0.7,
        top_p=0.95,
        max_tokens=512,
        n_threads=os.cpu_count(),
        verbose=False
    )

# --- Extract text from PDF ---
def load_pdf_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page_text := page.extract_text():
            text += page_text
    return text

# --- Chunk PDF text into LangChain documents ---
def split_text_to_documents(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

# --- Embed chunks and build FAISS vector store ---
def create_vectorstore(documents):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(documents, embedding_model)

# --- Streamlit App UI ---
st.set_page_config(page_title="Offline PDF Chat", layout="wide")
st.title("📄 Offline PDF Chat with Local LLM")

# Upload PDF
uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_pdf:
    with st.spinner("Processing PDF..."):
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_pdf.read())
            tmp_path = tmp_file.name

        # Process PDF
        text = load_pdf_text(tmp_path)
        documents = split_text_to_documents(text)
        vectorstore = create_vectorstore(documents)
        llm = load_llm()

        # Build retrieval chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        st.success("PDF is processed! You can now chat with it.")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display previous messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input box (like ChatGPT)
        if prompt := st.chat_input("Ask a question about the PDF..."):
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Thinking..."):
                result = qa_chain.run(prompt)

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(result)

            # Save to session state
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": result})
