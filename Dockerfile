FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MJPEG_PORT=8502 \
    MJPEG_STREAM_URL=/stream

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl \
    nginx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Remove GUI opencv-python pulled by ultralytics & force headless build + verify at build time
RUN pip uninstall -y opencv-python && \
    pip install --no-cache-dir --force-reinstall --no-deps "opencv-python-headless>=4.8.0" && \
    python -c "import cv2; print('OpenCV OK:', cv2.__version__)"

RUN useradd -m -u 1000 hazarduser && \
    mkdir -p /app /tmp/ultralytics /var/log/nginx /var/lib/nginx && \
    chown -R hazarduser:hazarduser /app /tmp/ultralytics /var/log/nginx /var/lib/nginx

COPY --chown=hazarduser:hazarduser . /app

# Strip Windows CRLF line endings so start.sh never fails on Linux
RUN sed -i 's/\r$//' /app/start.sh /app/nginx.conf && chmod +x /app/start.sh

USER hazarduser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=35s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["/app/start.sh"]
