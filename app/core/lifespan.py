"""
应用生命周期管理
"""
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.core.config import ModelConfig, ServiceConfig, ProviderConfig
from app.services.embedding_service import EmbeddingModel

logger = logging.getLogger(__name__)

# 全局变量存储模型
embedding_model = None


def get_embedding_model():
    """获取embedding模型实例"""
    global embedding_model
    return embedding_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global embedding_model
    
    logger.info("Initializing BGE embedding model...")
    start_time = time.time()
    
    try:
        embedding_model = EmbeddingModel(
            model_path=ModelConfig.MODEL_PATH,
            tokenizer_path=ModelConfig.TOKENIZER_PATH,
            providers=ProviderConfig.get_providers(),
            max_workers=ServiceConfig.EXECUTOR_MAX_WORKERS,
            tokenizer_executor_workers=ServiceConfig.TOKENIZER_EXECUTOR_WORKERS,
            inter_op_num_threads=ServiceConfig.ONNX_INTER_OP_NUM_THREADS,
            log_severity_level=ServiceConfig.ONNX_LOG_SEVERITY_LEVEL
        )
        logger.info(f"Model loaded in {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.exception(f"Model initialization failed: {str(e)}")
        raise
    
    yield
    
    # 清理资源
    if embedding_model is not None:
        del embedding_model
        embedding_model = None
        logger.info("Model resources released")

