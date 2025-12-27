"""
中间件
"""
import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


async def log_requests_middleware(request: Request, call_next):
    """请求日志中间件"""
    start_time = time.time()
    
    # 只记录embedding相关的请求
    if "/v1/embeddings" in str(request.url):
        logger.info("=" * 80)
        logger.info(" HTTP请求信息:")
        logger.info(f" 方法: {request.method}")
        logger.info(f" URL: {request.url}")
        logger.info(f" 客户端IP: {request.client.host if request.client else 'unknown'}")
        logger.info(f" 客户端端口: {request.client.port if request.client else 'unknown'}")
        
        # 记录重要的请求头
        important_headers = [
            'user-agent', 'content-type', 'content-length',
            'authorization', 'accept', 'accept-encoding'
        ]
        logger.info(" 请求头信息:")
        for header in important_headers:
            if header in request.headers:
                value = request.headers[header]
                # 隐藏敏感信息
                if header == 'authorization':
                    value = value[:30] + '...' if len(value) > 30 else value
                logger.info(f"  {header}: {value}")
        
        # 对于POST请求，记录请求体信息
        if request.method == "POST":
            try:
                body = await request.body()
                if body:
                    body_str = body.decode('utf-8')
                    logger.info(f" 请求体内容 (长度: {len(body)}):")
                    logger.info(f" {body_str[:500]}{'...' if len(body_str) > 500 else ''}")
                    
                    # 尝试解析JSON
                    try:
                        import json
                        parsed_json = json.loads(body_str)
                        logger.info(" JSON结构:")
                        logger.info(f" {json.dumps(parsed_json, indent=2, ensure_ascii=False)[:400]}{'...' if len(json.dumps(parsed_json, ensure_ascii=False)) > 400 else ''}")
                    except json.JSONDecodeError as e:
                        logger.warning(f" JSON解析失败: {e}")
                else:
                    logger.info(" 请求体为空")
            except Exception as e:
                logger.error(f" 无法读取请求体: {e}")
    
    # 处理请求
    response = await call_next(request)
    
    # 记录响应信息
    if "/v1/embeddings" in str(request.url):
        process_time = time.time() - start_time
        logger.info(f" 请求处理时间: {process_time:.3f}s")
        logger.info(f" 响应状态码: {response.status_code}")
        logger.info("=" * 80)
    
    return response

