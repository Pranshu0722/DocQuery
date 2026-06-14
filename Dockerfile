# Use Python 3.12
FROM python:3.12-slim

WORKDIR /app

# 1. Install system dependencies
#    - tesseract-ocr: required for image OCR
#    - build-essential: needed by some Python wheels
#    - curl: used for healthchecks
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy requirements first (so this layer is cached on code-only changes)
COPY requirements.txt .

# 3. Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the code
COPY . .

# 5. Tell the app it's running inside a container (used to switch Ollama URL)
ENV IS_DOCKER=true

# 6. Expose Streamlit port
EXPOSE 8501

# 7. Run the app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
