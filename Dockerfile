# Use Python 3.10
FROM python:3.10-slim

WORKDIR /app

# 1. Install System Dependencies
# 'tesseract-ocr' is the Linux equivalent of your .exe
# 'curl' is useful for health checks
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy requirements first (for caching)
COPY requirements.txt .

# 3. Install Python libraries
# We use the extra-index-url inside the command to ensure GPU torch is found
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the code
COPY . .

# 5. Set the Environment Variable so Python knows it's inside Docker
ENV IS_DOCKER=true

# 6. Expose Streamlit port
EXPOSE 8501

# 7. Run the app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]