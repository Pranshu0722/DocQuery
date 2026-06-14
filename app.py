import os
import uuid
import requests
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

def _format_size(num_bytes):
    if not num_bytes:
        return ""
    gb = num_bytes / (1024 ** 3)
    if gb >= 1:
        return f"{gb:.1f} GB"
    mb = num_bytes / (1024 ** 2)
    return f"{mb:.0f} MB"

def _pretty_name(raw_name):
    # Strip ":latest" since it's the default and adds noise
    name = raw_name.replace(":latest", "")
    # Title-case the model family (everything before ":")
    if ":" in name:
        family, tag = name.split(":", 1)
        return f"{family.title()} ({tag})"
    return name.title()

def list_ollama_models():
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        resp.raise_for_status()
        return resp.json().get("models", [])
    except Exception:
        return []

available_models = list_ollama_models()

if available_models:
    # Map pretty label -> raw model id
    label_to_id = {
        f"{_pretty_name(m['name'])} — {_format_size(m.get('size'))}".rstrip(" —"): m["name"]
        for m in available_models
    }
    labels = list(label_to_id.keys())

    # Pick a default label that matches OLLAMA_MODEL if present
    default_idx = 0
    for i, lbl in enumerate(labels):
        if label_to_id[lbl] == OLLAMA_MODEL:
            default_idx = i
            break

    selected_label = st.sidebar.selectbox("LLM Model", labels, index=default_idx)
    selected_model = label_to_id[selected_label]
else:
    st.sidebar.warning("Could not reach Ollama. Is it running?")
    selected_model = OLLAMA_MODEL
    st.sidebar.write(f"LLM: **{selected_model}** (fallback)")

if "active_model" not in st.session_state:
    st.session_state.active_model = selected_model

if selected_model != st.session_state.active_model:
    st.session_state.active_model = selected_model
    if st.session_state.get("qa") is not None and st.session_state.get("db") is not None:
        st.session_state.qa = None  # rebuilt below from cached db
        st.rerun()

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

def build_qa_chain(db, model_name):
    llm = OllamaLLM(model=model_name, temperature=0.3, base_url=OLLAMA_URL)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    return qa

# -----------------------------------------------------------
# UI
# -----------------------------------------------------------

st.title("Chat With File")
st.markdown("Upload a PDF, DOCX, TXT, or image and chat with it locally.")

if "qa" not in st.session_state:
    st.session_state.qa = None
if "db" not in st.session_state:
    st.session_state.db = None
if "history" not in st.session_state:
    st.session_state.history = []

# Rebuild the chain if the model changed but we still have a vector DB
if st.session_state.qa is None and st.session_state.db is not None:
    st.session_state.qa = build_qa_chain(st.session_state.db, selected_model)

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
                st.session_state.db = db
                st.session_state.qa = build_qa_chain(db, selected_model)
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

    for entry in st.session_state.history:
        role, msg = entry[0], entry[1]
        sources = entry[2] if len(entry) > 2 else None
        with st.chat_message(role):
            st.markdown(msg)
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for i, doc in enumerate(sources, 1):
                        st.markdown(f"**Chunk {i}**")
                        st.text(doc.page_content)

    if user_q := st.chat_input("Ask a question..."):
        st.session_state.history.append(("user", user_q))
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.spinner("Thinking..."):
            result = st.session_state.qa.invoke({"question": user_q})
            answer = result["answer"]
            sources = result.get("source_documents", [])

        st.session_state.history.append(("assistant", answer, sources))
        with st.chat_message("assistant"):
            st.markdown(answer)
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for i, doc in enumerate(sources, 1):
                        st.markdown(f"**Chunk {i}**")
                        st.text(doc.page_content)
