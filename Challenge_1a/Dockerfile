FROM --platform=linux/amd64 python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    paddlepaddle==2.5.1 \
    paddleocr==2.7.0.3 \
    transformers==4.35.0 \
    torch==2.1.0 \
    torchvision==0.16.0 \
    PyMuPDF==1.23.8 \
    Pillow==10.1.0 \
    numpy==1.24.3 \
    scikit-learn==1.3.2

# Copy the processing script
COPY process_pdfs.py .

# Run the script
CMD ["python", "process_pdfs.py"] 