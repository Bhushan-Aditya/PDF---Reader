#!/bin/bash

# Truly Dynamic OCR-Based PDF Structure Extractor
# Docker Runner Script

echo "🚀 Starting Truly Dynamic OCR-Based PDF Structure Extractor"
echo "================================================================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create input directory if it doesn't exist
if [ ! -d "input" ]; then
    echo "📁 Creating input directory..."
    mkdir -p input
fi

# Create output directory if it doesn't exist
if [ ! -d "output" ]; then
    echo "📁 Creating output directory..."
    mkdir -p output
fi

# Check if there are PDF files in input directory
if [ ! "$(ls -A input/*.pdf 2>/dev/null)" ]; then
    echo "⚠️  No PDF files found in input directory."
    echo "   Please place your PDF files in the 'input' directory."
    echo "   Example: cp your_file.pdf input/"
    exit 1
fi

echo "✅ Found PDF files in input directory"
echo "🔧 Building Docker image..."

# Build the Docker image
docker-compose build

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
    echo "🚀 Running OCR-based PDF structure extraction..."
    
    # Run the container
    docker-compose up --abort-on-container-exit
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 Processing completed successfully!"
        echo "📁 Results saved in 'output' directory"
        echo ""
        echo "📊 Generated files:"
        ls -la output/*.json 2>/dev/null || echo "   No JSON files found"
    else
        echo "❌ Processing failed. Check the logs above for errors."
        exit 1
    fi
else
    echo "❌ Failed to build Docker image. Check the logs above for errors."
    exit 1
fi 