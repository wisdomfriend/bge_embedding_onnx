#!/bin/bash
set -e

# 设置默认值
BGE_PORT=${BGE_PORT:-7979}
BGE_HOST=${BGE_HOST:-0.0.0.0}
BGE_LIMIT_CONCURRENCY=${BGE_LIMIT_CONCURRENCY:-1000}

# 设置 Uvicorn 协程并发限制
export UVICORN_LIMIT_CONCURRENCY=$BGE_LIMIT_CONCURRENCY

echo "Starting BGE Embedding Service..."
echo "Port: $BGE_PORT"
echo "Host: $BGE_HOST"
echo "Limit Concurrency: $BGE_LIMIT_CONCURRENCY"

# 检查模型文件是否存在
if [ -z "$BGE_MODEL_PATH" ]; then
    echo "Warning: BGE_MODEL_PATH not set, using default"
fi

if [ -z "$BGE_TOKENIZER_PATH" ]; then
    echo "Warning: BGE_TOKENIZER_PATH not set, using default"
fi

# 使用gunicorn启动服务（生产环境）
echo "Starting with gunicorn (production mode)..."
exec gunicorn -w $(nproc) \
    -k uvicorn.workers.UvicornWorker \
    --bind "$BGE_HOST:$BGE_PORT" \
    --timeout 300 \
    --access-logfile - \
    --error-logfile - \
    app.main:app

