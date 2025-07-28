# Truly Dynamic OCR-Based PDF Structure Extractor
# Docker image for Adobe India Hackathon 2025

FROM --platform=linux/amd64 python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OCR
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ocr_adaptive_processor.py .

# Create input and output directories
RUN mkdir -p input output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command to process PDFs
CMD ["python", "ocr_adaptive_processor.py"] 