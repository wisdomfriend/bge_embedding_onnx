"""
异常处理
"""
import logging
from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def not_found_handler(request: Request, exc):
    """404异常处理"""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": {
                "message": "Not Found",
                "type": "not_found_error",
                "path": str(request.url.path)
            }
        }
    )


async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.exception(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_server_error",
                "details": str(exc) if logger.level == logging.DEBUG else None
            }
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求验证异常处理"""
    logger.error("=" * 80)
    logger.error(" 请求验证失败!")
    logger.error(f" 请求URL: {request.url}")
    logger.error(f" 请求方法: {request.method}")
    logger.error(f" 客户端IP: {request.client.host if request.client else 'unknown'}")
    
    # 记录验证错误详情
    logger.error(" 验证错误详情:")
    for i, error in enumerate(exc.errors(), 1):
        logger.error(f"  错误 {i}:")
        logger.error(f"    字段路径: {error.get('loc', 'unknown')}")
        logger.error(f"    错误类型: {error.get('type', 'unknown')}")
        logger.error(f"    错误消息: {error.get('msg', 'unknown')}")
        logger.error(f"    输入值: {error.get('input', 'unknown')}")
    
    logger.error("=" * 80)
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "message": "Request validation failed",
                "type": "invalid_request_error",
                "details": exc.errors()
            }
        }
    )

