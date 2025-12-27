"""
健康检查端点
"""
import logging
from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.core.lifespan import get_embedding_model
from app.core.config import ProviderConfig

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=dict, tags=["health"])
async def root():
    """根路径"""
    return {
        "message": "BGE Embedding ONNX API Service",
        "version": "1.0.0",
        "docs": "/docs"
    }


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """健康检查"""
    model = get_embedding_model()
    
    if model is None:
        return HealthResponse(
            status="unhealthy",
            model="bge-large-zh-v1.5",
            model_version="1.5",
            providers=[],
            environment="unknown"
        )
    
    providers = []
    if model.session:
        providers = model.session.get_providers()
    
    return HealthResponse(
        status="healthy",
        model="bge-large-zh-v1.5",
        model_version="1.5",
        providers=providers,
        environment="production"
    )

