# ----------------------------------------------------
# Base Python image
# ----------------------------------------------------
FROM python:3.10-slim

# ----------------------------------------------------
# System dependencies (POPPLER for PDF → PNG)
# ----------------------------------------------------
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------
# Create working directory
# ----------------------------------------------------
WORKDIR /app

# ----------------------------------------------------
# Copy project files
# ----------------------------------------------------
# requirements first to leverage Docker layer caching
COPY requirements.txt .

# install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# copy ALL source code (including data_pipeline, inference.py, example_provider)
COPY . .

# ensure data_pipeline is importable
ENV PYTHONPATH="/app:${PYTHONPATH}"

# ----------------------------------------------------
# Runtime arguments (optional defaults)
# ----------------------------------------------------
ENV GOOGLE_CLOUD_PROJECT="mlops-472423"
ENV VERTEX_LOCATION="us-central1"
ENV VERTEX_MODEL="gemini-2.5-flash"
ENV VERTEX_TEMPERATURE="0.0"
ENV VERTEX_MAX_TOKENS="2048"

# ----------------------------------------------------
# Command to run inference
# ----------------------------------------------------
# You can overwrite this during `docker run`
CMD ["python", "inference.py"]
