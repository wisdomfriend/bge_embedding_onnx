"""
API路由注册
"""
from fastapi import APIRouter
from app.api.endpoints import health, embeddings

# 创建路由
api_router = APIRouter()

# 注册健康检查路由
api_router.add_api_route("/", health.root, methods=["GET"], tags=["health"])
api_router.add_api_route("/health", health.health_check, methods=["GET"], tags=["health"])

# 注册Embedding路由
api_router.add_api_route("/v1/embeddings", embeddings.create_embeddings, methods=["POST"], tags=["embeddings"])

