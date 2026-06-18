FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application code
COPY . .

# Download ONNX model for lightweight inference
RUN python download_model.py

# Expose port (Render uses $PORT env var)
EXPOSE 5000

# Use ENV so $PORT has a default fallback
ENV PORT=5000
ENV LITE_MODE=1

# Start with gunicorn - must use shell form so $PORT expands
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 300 --preload app:app
