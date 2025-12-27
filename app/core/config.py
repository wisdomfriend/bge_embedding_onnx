"""
应用配置文件
"""
import os
import platform
from pathlib import Path
from typing import Optional

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 模型配置
class ModelConfig:
    """模型配置"""
    # 自动判断操作系统，设置模型路径
    if platform.system().lower().startswith("win"):
        MODEL_PATH = os.getenv("BGE_MODEL_PATH", str(BASE_DIR / "bge-large-zh-v1.5.onnx"))
        TOKENIZER_PATH = os.getenv(
            "BGE_TOKENIZER_PATH", 
            r"D:\backup\models\BAAI\bge-large-zh-v1___5"
        )
    else:
        MODEL_PATH = os.getenv(
            "BGE_MODEL_PATH",
            "/embeddings/model/bge-large-zh-v1.5-onnx/bge-large-zh-v1.5.onnx"
        )
        TOKENIZER_PATH = os.getenv(
            "BGE_TOKENIZER_PATH",
            "/embeddings/model/bge-large-zh-v1.5-onnx"
        )


# 服务配置
class ServiceConfig:
    """服务配置"""
    # 批量处理配置
    MAX_BATCH_SIZE = int(os.getenv("BGE_MAX_BATCH_SIZE", "32"))
    MAX_BATCH_SIZE_ALL = int(os.getenv("BGE_MAX_BATCH_SIZE_ALL", "256"))
    
    # 线程池配置
    # 推理 executor：GPU推理时设置为1,通过works来控制并发数
    EXECUTOR_MAX_WORKERS = int(os.getenv("BGE_EXECUTOR_WORKERS", "1"))
    # Tokenizer executor：用于 tokenize 操作
    TOKENIZER_EXECUTOR_WORKERS = int(os.getenv("BGE_TOKENIZER_EXECUTOR_WORKERS", "4"))
    
    # 模型推理配置 bge-embedding最大支持512个token
    MAX_SEQUENCE_LENGTH = int(os.getenv("BGE_MAX_SEQUENCE_LENGTH", "512"))
    
    # ONNX Runtime配置
    ONNX_INTER_OP_NUM_THREADS = int(os.getenv("BGE_ONNX_INTER_OP_THREADS", "1"))
    ONNX_LOG_SEVERITY_LEVEL = int(os.getenv("BGE_ONNX_LOG_LEVEL", "3"))


# 服务器配置
class ServerConfig:
    """服务器配置"""
    HOST = os.getenv("BGE_HOST", "0.0.0.0")
    PORT = int(os.getenv("BGE_PORT", "8080"))
    RELOAD = os.getenv("BGE_RELOAD", "false").lower() == "true"
    WORKERS = int(os.getenv("BGE_WORKERS", "1"))
    TIMEOUT_KEEP_ALIVE = int(os.getenv("BGE_TIMEOUT_KEEP_ALIVE", "300"))
    LOG_LEVEL = os.getenv("BGE_LOG_LEVEL", "INFO")


# ONNX Runtime Provider配置
class ProviderConfig:
    """ONNX Runtime Provider配置"""
    @staticmethod
    def get_providers():
        """获取可用的执行提供者"""
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        
        # 优先使用ROCM（AMD GPU），否则使用CPU
        if 'ROCMExecutionProvider' in available_providers:
            return [('ROCMExecutionProvider', {'device_id': 0})]
        elif 'CUDAExecutionProvider' in available_providers:
            return [('CUDAExecutionProvider', {'device_id': 0})]
        else:
            return ['CPUExecutionProvider']

