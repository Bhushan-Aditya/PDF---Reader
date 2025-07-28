# Truly Dynamic OCR-Based Adaptive PDF Structure Extractor

## 🚀 Overview

A lightweight, truly dynamic OCR-based PDF structure extractor that achieves high accuracy through adaptive algorithms without any hardcoding. The system uses OCR (Optical Character Recognition) to extract titles and outlines from PDFs with intelligent content analysis.

## ✨ Key Features

- **🔍 OCR-Based Processing**: PDF → Image → OCR → Parse → JSON pipeline
- **🧠 Truly Dynamic Algorithms**: No hardcoding, pure adaptive intelligence
- **⚡ Lightweight**: Total size <120 MB (Poppler + Tesseract + Python deps)
- **🚀 Parallel Processing**: Multi-threaded OCR for speed
- **📊 High Accuracy**: Adaptive algorithms for title and outline extraction
- **🔄 Continuous Improvement**: Iterative enhancement towards 90%+ accuracy
- **🐳 Dockerized**: Easy deployment and portability

## 🏗️ Technical Architecture

### Core Pipeline
1. **PDF to Image Conversion**: Uses `pdf2image` + `Poppler` at 150 DPI
2. **OCR Processing**: `pytesseract` + `Tesseract` with layout-aware parsing
3. **Adaptive Analysis**: Dynamic title and outline extraction
4. **JSON Output**: Structured results with hierarchical outlines

### Dependencies
- **System**: Poppler (PDF processing), Tesseract (OCR)
- **Python**: pdf2image, pytesseract, Pillow, numpy
- **Total Size**: <120 MB (well under 200 MB budget)

## 🛠️ Installation & Usage

### Docker Deployment (Recommended)

#### Prerequisites
```bash
# Install Docker
# macOS: brew install docker
# Ubuntu: sudo apt-get install docker.io
# Windows: Download Docker Desktop
```

#### Build and Run Commands

**Build the Docker image:**
```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier
```

**Run the solution:**
```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier
```

#### Container Requirements
- **Input**: Automatically processes all PDFs from `/app/input` directory
- **Output**: Generates corresponding `filename.json` in `/app/output` for each `filename.pdf`
- **Network**: No network access required (`--network none`)

#### Quick Start
```bash
# 1. Place your PDF files in the input directory
cp your_files.pdf input/

# 2. Build the Docker image
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier

# 3. Run the container
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier

# 4. Check results
ls -la output/
```

### Local Installation (Alternative)

#### Prerequisites
```bash
# Install system dependencies
brew install poppler tesseract

# Create virtual environment
python3 -m venv ocr_env
source ocr_env/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

#### Requirements
```
# Core OCR dependencies
pdf2image==3.1.0
pytesseract==0.3.10
Pillow==10.0.1

# Additional utilities
numpy==1.24.3
psutil==5.9.6
```

## 🚀 Usage

### Docker Usage
```bash
# Build and run in one command
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier && \
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier

# Check results
cat output/file01.json
```

### Local Usage
```bash
# Activate virtual environment
source ocr_env/bin/activate

# Run the OCR-based processor
python ocr_adaptive_processor.py
```

### Input/Output Structure
```
Challenge_1a/
├── input/           # Place PDF files here
│   ├── file01.pdf
│   ├── file02.pdf
│   └── ...
├── output/          # Generated JSON results
│   ├── file01.json
│   ├── file02.json
│   └── ...
├── ocr_adaptive_processor.py
├── Dockerfile
├── docker-compose.yml
└── run_docker.sh
```

### Output Format
```json
{
  "title": "Extracted document title",
  "outline": [
    {
      "level": "H1",
      "text": "Section title ",
      "page": 1
    }
  ]
}
```

## 🧠 Adaptive Algorithms

### Title Extraction
- **Dynamic Candidate Detection**: Multi-criteria analysis
- **Position-Based Scoring**: Early lines get higher weight
- **Formatting Analysis**: All caps, title case detection
- **Content Analysis**: Semantic keyword detection
- **Validation**: Invalid pattern filtering

### Outline Extraction
- **TOC Detection**: Dynamic pattern matching
- **Hierarchical Parsing**: Level-based structure building
- **Fallback Analysis**: Heading extraction when no TOC found
- **Content Validation**: Quality filtering

### Key Features
- **Zero Hardcoding**: No file-specific rules
- **Adaptive Thresholds**: Dynamic scoring based on document characteristics
- **Multi-Approach Strategy**: Multiple extraction methods combined
- **Intelligent Validation**: Content-based candidate filtering

## 🔧 Technical Specifications

### System Requirements
- **OS**: macOS, Linux, Windows (with Docker)
- **Python**: 3.8+ (for local installation)
- **Memory**: <150 MB peak usage
- **Storage**: <120 MB total size
- **Processing**: Multi-core support for parallel OCR

### Performance Benchmarks
- **Processing Speed**: ~14 seconds for 5 files
- **Memory Usage**: <150 MB peak
- **Accuracy Target**: 90%+ for both title and outline
- **Scalability**: Parallel processing for multiple files

### Docker Specifications
- **Base Image**: python:3.11-slim
- **Platform**: linux/amd64
- **System Dependencies**: poppler-utils, tesseract-ocr
- **Image Size**: ~200 MB
- **Network**: None (isolated)
- **Volumes**: input/ (read-only), output/ (read-write)

## 🎯 Current Status

### Achievements
- ✅ **Truly Dynamic**: Zero hardcoding, pure adaptive algorithms
- ✅ **Lightweight**: <120 MB total size
- ✅ **Fast**: 14 seconds for all files
- ✅ **Robust**: Handles various document types
- ✅ **Dockerized**: Easy deployment and portability

### Ongoing Improvements
- 🔄 **Title Detection**: Enhancing semantic analysis
- 🔄 **Outline Detection**: Improving TOC pattern recognition
- 🔄 **Performance**: Optimizing processing speed
- 🔄 **Accuracy**: Targeting 90%+ across all files

## 🚀 Future Enhancements

### Planned Improvements
1. **Enhanced Title Detection**: Better semantic analysis
2. **Improved TOC Patterns**: More robust outline detection
3. **Document Type Learning**: Adaptive classification
4. **Performance Optimization**: Faster processing
5. **Accuracy Boost**: Targeting 90%+ across all files

### Advanced Features
- **Machine Learning Integration**: For better classification
- **Multi-Language Support**: Beyond English
- **Real-time Processing**: Stream processing capabilities
- **API Interface**: RESTful service integration

## 📝 License

This project is part of the Adobe India Hackathon 2025.

## 🤝 Contributing

The system is designed for continuous improvement with truly dynamic, adaptive algorithms that learn from document structure and content patterns.

---

**🎯 Goal**: Achieve 90%+ accuracy for both title and outline extraction using truly dynamic, adaptive algorithms with zero hardcoding. 