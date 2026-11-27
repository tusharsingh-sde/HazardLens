FROM python:3.12-slim

# Environment optimizations
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# System dependencies for OpenCV, FFmpeg, and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install CPU-only PyTorch BEFORE requirements/ultralytics to keep image lean
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create dedicated non-root user and ensure directory permissions
RUN useradd -m -u 1000 hazarduser && \
    mkdir -p /app /tmp/ultralytics && \
    chown -R hazarduser:hazarduser /app /tmp/ultralytics

# Copy application files with non-root ownership
COPY --chown=hazarduser:hazarduser . /app

# Switch to non-root execution
USER hazarduser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
