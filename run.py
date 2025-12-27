#!/usr/bin/env python3
"""
BGE Embedding服务启动脚本
用于开发环境和测试环境启动服务
支持热重载和调试功能
"""
import os
import logging
from app.main import app
from app.core.logging_config import setup_logging
from app.core.config import ServerConfig
import uvicorn

# 配置日志
setup_logging(ServerConfig.LOG_LEVEL)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("Starting BGE Embedding Service in Development/Testing Mode")
    logger.info(f"Host: {ServerConfig.HOST}")
    logger.info(f"Port: {ServerConfig.PORT}")
    logger.info(f"Log Level: {ServerConfig.LOG_LEVEL}")
    logger.info(f"Reload (Hot Reload): {ServerConfig.RELOAD}")
    logger.info(f"Timeout Keep Alive: {ServerConfig.TIMEOUT_KEEP_ALIVE}s")
    logger.info("Note: Use uvicorn for development (supports hot reload and debugging)")
    logger.info("Note: Workers not used in development mode (single process for debugging)")
    logger.info("=" * 50)
    
    uvicorn.run(
        "app.main:app",
        host=ServerConfig.HOST,
        port=ServerConfig.PORT,
        reload=ServerConfig.RELOAD,
        loop="asyncio",
        timeout_keep_alive=ServerConfig.TIMEOUT_KEEP_ALIVE,
        log_level=ServerConfig.LOG_LEVEL.lower(),
        access_log=True,
    )

