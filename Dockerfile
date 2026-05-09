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

# Copy your full requirements.txt
COPY requirements.txt .

# Install all your dependencies (pip will resolve conflicts)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p temp

EXPOSE 8000

CMD ["uvicorn", "simple_backend:app", "--host", "0.0.0.0", "--port", "8000"]