FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (skip torch if too large)
RUN pip install --no-cache-dir \
    fastapi==0.136.1 \
    uvicorn==0.46.0 \
    python-multipart==0.0.27 \
    python-dotenv==1.2.2 \
    anthropic==0.100.0 \
    easyocr==1.7.2 \
    opencv-python-headless==4.13.0.92 \
    numpy==1.26.4 \
    pillow==12.2.0

# Copy application
COPY . .

# Create temp directory
RUN mkdir -p temp

EXPOSE 8000

CMD ["uvicorn", "simple_backend:app", "--host", "0.0.0.0", "--port", "8000"]