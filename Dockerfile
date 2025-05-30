# Stage 1: Builder
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

# Install system dependencies for OpenCV, Torch, and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools for modern Python packaging
RUN pip install --upgrade pip setuptools

# Copy requirements and install Python dependencies (user-level for portability)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime image
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install minimal runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade setuptools again (sometimes needed for some packages)
RUN pip install --upgrade setuptools

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create required directories at build time to avoid runtime errors
RUN mkdir -p temp_uploads visualizations cache

# Expose FastAPI port
EXPOSE 8080

# Set environment variables for production (optional)
ENV PYTHONUNBUFFERED=1

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
