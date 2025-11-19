FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python + pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy your app code
# COPY app/ ./app/
COPY . .

EXPOSE 8080

# Start FastAPI via Uvicorn
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]
