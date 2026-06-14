import os
import uuid
import streamlit as st
import torch
from dotenv import load_dotenv
from PIL import Image
import pytesseract

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
# SETTINGS
# -----------------------------------------------------------

load_dotenv()

# Tesseract OCR path
#  - On Windows, set TESSERACT_PATH in .env if Tesseract isn't on PATH
#  - On Linux/Docker, the binary is on PATH as `tesseract`
default_tesseract = "tesseract" if os.name != "nt" else None
TESSERACT_PATH = os.getenv("TESSERACT_PATH", default_tesseract)
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Ollama config — host.docker.internal lets a container reach the host
if os.getenv("IS_DOCKER"):
    DEFAULT_OLLAMA_URL = "http://host.docker.internal:11434"
else:
    DEFAULT_OLLAMA_URL = "http://localhost:11434"
OLLAMA_URL = os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")

DB_BASE_PATH = "./db"
os.makedirs(DB_BASE_PATH, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Chat With File", layout="wide")
st.sidebar.title("Configuration")
st.sidebar.write(f"Processing on: **{DEVICE.upper()}**")
st.sidebar.write(f"LLM: **{OLLAMA_MODEL}**")

# -----------------------------------------------------------
# HELPERS
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

        elif file_type in ["png", "jpg", "jpeg"]:
            image = Image.open(uploaded_file)
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
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.3, base_url=OLLAMA_URL)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        memory=memory
    )
    return qa

# -----------------------------------------------------------
# UI
# -----------------------------------------------------------

st.title("Chat With File")
st.markdown("Upload a PDF, DOCX, TXT, or image and chat with it locally.")

if "qa" not in st.session_state:
    st.session_state.qa = None
if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader(
    "Upload Document or Image",
    type=["txt", "pdf", "docx", "png", "jpg", "jpeg"]
)

if uploaded_file and st.session_state.qa is None:
    with st.spinner(f"Processing on {DEVICE}..."):
        text = load_text_from_file(uploaded_file)

        if text:
            chunks = split_into_chunks(text)
            if chunks:
                db = create_vector_db(chunks)
                st.session_state.qa = build_qa_chain(db)
                st.success(f"Loaded {len(chunks)} chunks! Ready to chat.")
            else:
                st.error("File is empty or could not be read.")
        else:
            st.error("Could not extract text. If this is an image, make sure Tesseract is installed.")

if st.session_state.qa:
    if st.sidebar.button("Clear Conversation"):
        st.session_state.history = []
        st.session_state.qa.memory.clear()
        st.rerun()

    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

    if user_q := st.chat_input("Ask a question..."):
        st.session_state.history.append(("user", user_q))
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.spinner("Thinking..."):
            result = st.session_state.qa.invoke({"question": user_q})
            answer = result["answer"]

        st.session_state.history.append(("assistant", answer))
        with st.chat_message("assistant"):
            st.markdown(answer)
