#!/bin/sh
set -e

# Start Nginx reverse proxy on port 8501
nginx -c /app/nginx.conf -e /dev/stderr

# Run Streamlit in foreground on internal port 8503 (overrides config.toml port)
exec streamlit run app.py \
    --server.port=8503 \
    --server.address=127.0.0.1 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
