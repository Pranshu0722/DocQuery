import os
import uuid
import streamlit as st
import torch
from PIL import Image # For handling images
import pytesseract    # For OCR (Extracting text from images)

# PDF / DOCX reading
from pypdf import PdfReader
import docx2txt

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# -----------------------------------------------------------
# ✅ SETTINGS & GPU SETUP
# -----------------------------------------------------------

# 1. SMART TESSERACT PATH 
# If running on Windows (nt), use your D: drive path.
# If running on Docker (posix), use the default Linux path.
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'D:\tesseract(img OCR)\tesseract.exe'
else:
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'

# 2. SMART OLLAMA URL
# Docker containers cannot see "localhost" directly. They need a special bridge.
# If running in Docker (we will set an env variable later), use the bridge address.
if os.getenv('IS_DOCKER'):
    OLLAMA_URL = "http://host.docker.internal:11434"
else:
    OLLAMA_URL = "http://localhost:11434"

DB_BASE_PATH = "./db"
os.makedirs(DB_BASE_PATH, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Chat With File", layout="wide")
st.sidebar.title("Configuration")
st.sidebar.write(f"📌 Processing on: **{DEVICE.upper()}**")

# -----------------------------------------------------------
# 🔥 HELPER FUNCTIONS
# -----------------------------------------------------------

def load_text_from_file(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1].lower()
    text = ""

    try:
        if file_type == "txt":
            text = uploaded_file.read().decode("utf-8", errors="ignore")

        elif file_type == "pdf":
            pdf = PdfReader(uploaded_file)
            pages = [page.extract_text() or "" for page in pdf.pages]
            text = "\n".join(pages)

        elif file_type == "docx":
            text = docx2txt.process(uploaded_file)

        # 📸 NEW: IMAGE SUPPORT
        elif file_type in ["png", "jpg", "jpeg"]:
            image = Image.open(uploaded_file)
            # Extract text using Tesseract OCR
            text = pytesseract.image_to_string(image)
        
        else:
            st.error("Unsupported file type!")
            return None

    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

    return text

def split_into_chunks(text):
    if not text or not text.strip():
        return []
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

def create_vector_db(chunks):
    # ✅ GPU FIX: Using the DEVICE variable here
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE}  
    )

    unique_db_path = os.path.join(DB_BASE_PATH, f"session_{uuid.uuid4()}")
    os.makedirs(unique_db_path, exist_ok=True)

    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=unique_db_path
    )
    return db

def build_qa_chain(db):
    # Pass the smart URL here
    llm = OllamaLLM(model="tinyllama", temperature=0.3, base_url=OLLAMA_URL)
    # ... rest of the function remains the same ...

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    # PREDEFINED RAG FUNCTION
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,                           # BRAIN - LLM MODEL
        retriever=db.as_retriever(),       # DATA - VECTOR DB
        memory=memory
    )
    return qa

# -----------------------------------------------------------
# 🚀 STREAMLIT UI
# -----------------------------------------------------------

st.title("📚 Chat With File (GPU + Images)")
st.markdown("Upload a PDF, Docx, or Image to chat with it.")

# Session State
if "qa" not in st.session_state:
    st.session_state.qa = None
if "history" not in st.session_state:
    st.session_state.history = []

# File Uploader
uploaded_file = st.file_uploader(
    "Upload Document or Image",
    type=["txt", "pdf", "docx", "png", "jpg", "jpeg"]
)

# Process File
if uploaded_file and st.session_state.qa is None:
    with st.spinner(f"Processing on {DEVICE}..."):
        text = load_text_from_file(uploaded_file)
        
        if text:
            chunks = split_into_chunks(text)
            if chunks:
                db = create_vector_db(chunks)
                st.session_state.qa = build_qa_chain(db)
                st.success(f"✅ Loaded {len(chunks)} chunks! Ready to chat.")
            else:
                st.error("File is empty or could not be read.")
        else:
            st.error("Could not extract text. If this is an image, make sure Tesseract is installed.")

# Chat Interface
if st.session_state.qa:
    # Clear button
    if st.sidebar.button("🧹 Clear Conversation"):
        st.session_state.history = []
        st.session_state.qa.memory.clear()
        st.rerun()

    # Display History
    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

    # Input
    if user_q := st.chat_input("Ask a question..."):
        st.session_state.history.append(("user", user_q))
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.spinner("Thinking..."):
            # Add .invoke to call the chain
            result = st.session_state.qa.invoke({"question": user_q})
            answer = result["answer"]
        
        st.session_state.history.append(("assistant", answer))
        with st.chat_message("assistant"):
            st.markdown(answer)