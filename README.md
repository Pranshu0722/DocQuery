# Chat With File — Local RAG App

A fully offline chatbot that lets you upload **PDFs, Word docs, text files, or images** and ask questions about them. Powered by a local LLM via [Ollama](https://ollama.com/), [LangChain](https://www.langchain.com/) for retrieval, [ChromaDB](https://www.trychroma.com/) for vector storage, and [Streamlit](https://streamlit.io/) for the UI.

No API keys. No cloud calls. Your documents never leave your machine.

## Features

- Chat with **PDF, DOCX, TXT, and image** files (PNG/JPG/JPEG)
- Image support via **Tesseract OCR**
- Uses **Mistral / TinyLlama / Phi3** or any other Ollama model
- **GPU acceleration** when a CUDA-capable NVIDIA GPU is available, automatic fallback to CPU
- Conversational memory — follow-up questions are aware of context
- Per-session vector DB so multiple uploads stay isolated

## Tech Stack

| Layer | Tool |
|------|------|
| UI | Streamlit |
| LLM Runtime | Ollama (local) |
| Orchestration | LangChain |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | ChromaDB |
| OCR | Tesseract |
| Doc parsing | pypdf, docx2txt |

## Prerequisites

1. **Python 3.10–3.12**
2. **[Ollama](https://ollama.com/download)** installed and running
3. **[Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)** (only required for image uploads)

## Setup

### 1. Clone and enter the project
```bash
git clone <your-repo-url>
cd Local_LLM-main
```

### 2. Create a virtual environment
The fastest way is using [uv](https://docs.astral.sh/uv/) (recommended):
```bash
uv venv --python 3.12
.\.venv\Scripts\activate    # Windows
source .venv/bin/activate   # macOS/Linux
uv pip install -r requirements.txt
```

Or with standard pip:
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Pull an Ollama model
```bash
ollama pull tinyllama   # ~640 MB, lightest
# or
ollama pull mistral     # ~4 GB, best quality (needs ~5 GB free RAM)
# or
ollama pull phi3:mini   # ~2.3 GB, good balance
```

Make sure the model name in `app.py` (`build_qa_chain`) matches the one you pulled.

### 4. (Optional) Configure Tesseract path
If you want image OCR support, install Tesseract and update the path in `app.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r'<your-tesseract-path>\tesseract.exe'
```

## Run

Make sure Ollama is running in the background, then:
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

## Usage

1. Upload a document or image from the main page
2. Wait for the embedding step to finish (`✅ Loaded N chunks!`)
3. Ask questions in the chat box
4. Use **🧹 Clear Conversation** in the sidebar to reset memory

## Docker

A `Dockerfile` and `docker-compose.yml` are included so you can run the app in a container instead of setting up Python locally.

### Prerequisites

1. **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** installed and running
2. **Ollama** installed and running **on the host machine** (not inside the container)
3. The model you want to use already pulled on the host: `ollama pull tinyllama`

### Build and run

From the project root:
```bash
docker compose up --build
```

First build takes 5–10 minutes (downloading PyTorch, Tesseract, etc.). Subsequent runs are instant.

Open **http://localhost:8501** in your browser.

To stop:
```bash
docker compose down
```

### How it connects to Ollama

The container reaches Ollama on the host via `host.docker.internal:11434`. The app automatically switches to this URL when it detects it's running inside Docker (via the `IS_DOCKER` env variable).

### GPU support (optional)

If you have an NVIDIA GPU and want CUDA acceleration:

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host
2. Uncomment the `deploy:` block in `docker-compose.yml`
3. Swap the CPU torch lines in `requirements.txt` for CUDA versions:
   ```
   --extra-index-url https://download.pytorch.org/whl/cu124
   torch==2.6.0+cu124
   torchvision==0.21.0+cu124
   torchaudio==2.6.0+cu124
   ```
4. Rebuild: `docker compose up --build`

### Editing code without rebuilding

The `volumes` mount in `docker-compose.yml` syncs your local folder into the container. Edits to `app.py` are reflected immediately — Streamlit will auto-reload. You only need to rebuild when `requirements.txt` or the `Dockerfile` itself changes.

## Notes

- First startup is slow (~30–90 s) — PyTorch and embedding models are loaded into memory
- Large models like Mistral need significant RAM (5+ GB free). Use smaller models on low-RAM systems
- Vector databases are stored per-session in `./db/` and are gitignored

## Project Structure

```
Local_LLM-main/
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container image definition
├── docker-compose.yml  # Container orchestration
├── .gitignore
└── README.md
```

## License

MIT
