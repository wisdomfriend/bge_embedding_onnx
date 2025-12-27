"""
API数据模型（Pydantic Schemas）
"""
from typing import List, Union, Optional
from pydantic import BaseModel, validator


class EmbeddingRequest(BaseModel):
    """Embedding请求模型"""
    input: Union[str, List[str]]
    model: str = "bge-large-zh-v1.5"
    encoding_format: Optional[str] = None
    dimensions: Optional[int] = None
    user: Optional[str] = None
    
    @validator('input')
    def validate_input(cls, v):
        if isinstance(v, str):
            return v
        elif isinstance(v, list):
            if len(v) == 0:
                raise ValueError("Input list cannot be empty")
            if all(isinstance(item, str) for item in v):
                return v
            raise ValueError("Input must be a string or an array of strings")
        raise ValueError("Invalid input type")
    
    @validator('encoding_format')
    def validate_encoding_format(cls, v):
        if v is not None and v not in ["float", "base64"]:
            raise ValueError("encoding_format must be 'float' or 'base64'")
        return v
    
    @validator('dimensions')
    def validate_dimensions(cls, v):
        if v is not None and (v <= 0 or v > 4096):
            raise ValueError("dimensions must be a positive integer not exceeding 2048")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "input": "这是一个测试文本",
                "model": "bge-large-zh-v1.5",
                "encoding_format": None
            }
        }


class EmbeddingData(BaseModel):
    """单个Embedding数据"""
    object: str = "embedding"
    index: int
    embedding: Union[List[float], str]  # 可以是列表或base64编码的字符串
    
    class Config:
        schema_extra = {
            "example": {
                "object": "embedding",
                "index": 0,
                "embedding": [0.1, 0.2, 0.3, ...]
            }
        }


class EmbeddingUsage(BaseModel):
    """Token使用统计"""
    prompt_tokens: int
    total_tokens: int
    
    class Config:
        schema_extra = {
            "example": {
                "prompt_tokens": 10,
                "total_tokens": 10,
                "processing_time": 0.123
            }
        }


class EmbeddingResponse(BaseModel):
    """Embedding响应模型"""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage
    
    class Config:
        schema_extra = {
            "example": {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [0.1, 0.2, 0.3, ...]
                    }
                ],
                "model": "bge-large-zh-v1.5",
                "usage": {
                    "prompt_tokens": 10,
                    "total_tokens": 10
                }
            }
        }


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    model: str
    model_version: str
    providers: List[str]
    environment: str
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model": "bge-large-zh-v1.5",
                "model_version": "1.5",
                "providers": ["CPUExecutionProvider"],
                "environment": "production"
            }
        }

