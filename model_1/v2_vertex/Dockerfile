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
COPY model_1/v2_vertex/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy code + data dependencies needed by inference/Streamlit
COPY data-pipeline /app/data-pipeline
COPY model_1/v2_vertex /app/model_1/v2_vertex

# Ensure data_pipeline is importable
ENV PYTHONPATH="/app/model_1/v2_vertex:/app/data-pipeline:${PYTHONPATH}"
WORKDIR /app/model_1/v2_vertex

# ----------------------------------------------------
# Runtime arguments (optional defaults)
# ----------------------------------------------------
ENV GOOGLE_CLOUD_PROJECT="mlops-472423"
ENV VERTEX_LOCATION="us-central1"
ENV VERTEX_MODEL="gemini-2.5-flash"
ENV VERTEX_TEMPERATURE="0.0"
ENV VERTEX_MAX_TOKENS="2048"
ENV VERTEX_EXAMPLE_COUNT="3"
ENV DOC_TYPE="invoices"
ENV SAMPLE_PATH="/app/data-pipeline/data/unstructured/invoices"

# Streamlit defaults
ENV STREAMLIT_PORT="8501"
EXPOSE 8501

# ----------------------------------------------------
# Command to run Streamlit UI (default) or CLI inference
# ----------------------------------------------------
# Switch to CLI by setting APP_MODE=inference
CMD ["sh", "-c", "if [ \"${APP_MODE:-streamlit}\" = \"streamlit\" ]; then streamlit run streamlit_app.py --server.port=${PORT:-$STREAMLIT_PORT} --server.address=0.0.0.0; else python inference.py; fi"]
