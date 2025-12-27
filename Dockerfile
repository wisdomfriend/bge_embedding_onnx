FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn

# Copy application code
COPY app/ /app/app/
COPY run.py /app/
COPY scripts/ /app/scripts/

# Create directories for logs
RUN mkdir -p /app/logs

# Copy entrypoint script
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Expose port (default 8080, can be overridden via BGE_PORT env var)
ARG BGE_PORT=8080
EXPOSE ${BGE_PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD sh -c 'BGE_PORT=${BGE_PORT:-8080}; python -c "import requests; requests.get(\"http://localhost:${BGE_PORT}/health\", timeout=5)"' || exit 1

# Use entrypoint script to handle configurable port
ENTRYPOINT ["/app/docker-entrypoint.sh"]

