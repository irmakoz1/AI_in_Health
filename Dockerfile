FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OCR and computer vision
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    libpq-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p temp
RUN mkdir -p pipeline

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "simple_backend:app", "--host", "0.0.0.0", "--port", "8000"]