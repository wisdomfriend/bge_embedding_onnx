"""
Embedding服务
"""
import os
import time
import numpy as np
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """BGE Embedding模型封装"""
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        providers: List[Any],
        max_workers: int = 1,
        tokenizer_executor_workers: int = 4,
        inter_op_num_threads: int = 1,
        log_severity_level: int = 3
    ):
        """
        初始化模型
        
        Args:
            model_path: ONNX模型文件路径
            tokenizer_path: Tokenizer目录路径
            providers: ONNX Runtime执行提供者列表
            max_workers: 推理线程池最大工作线程数（用于 GPU 推理，建议为 1）
            tokenizer_executor_workers: Tokenizer 线程池最大工作线程数（CPU 操作，可以更大）
            inter_op_num_threads: ONNX Runtime内部操作线程数
            log_severity_level: ONNX Runtime日志级别
        """
        self.tokenizer = None
        self.pad_token_id = 0
        self.session = None
        # 推理 executor：用于 GPU 推理，控制并发数
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        # Tokenizer executor：用于 tokenize 操作，可以设置更大的并发数
        self.tokenizer_executor = ThreadPoolExecutor(max_workers=tokenizer_executor_workers)
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        
        self._load_model(providers, inter_op_num_threads, log_severity_level)
    
    def _load_model(
        self,
        providers: List[Any],
        inter_op_num_threads: int,
        log_severity_level: int
    ):
        """加载模型和tokenizer"""
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=True
            )
            self.pad_token_id = self.tokenizer.pad_token_id or 0
            logger.info(f"Tokenizer loaded from {self.tokenizer_path}")
            logger.info(f"Vocabulary size: {self.tokenizer.vocab_size}")
            
            # 检查模型文件是否存在
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            # 创建ONNX Runtime会话
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = log_severity_level
            sess_options.intra_op_num_threads = inter_op_num_threads
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )
            logger.info(f"ONNX Runtime session created")
            logger.info(f"Available providers: {self.session.get_providers()}")
            
        except Exception as e:
            logger.exception(f"Model loading failed: {str(e)}")
            # 确保在失败时清理资源
            self.tokenizer = None
            self.session = None
            raise RuntimeError(f"Model initialization failed: {str(e)}")
    
    def tokenize(
        self,
        texts: List[str],
        max_length: int = 512,
        return_tensors: str = "np"
    ) -> Dict[str, np.ndarray]:
        """
        Tokenize文本
        
        Args:
            texts: 文本列表
            max_length: 最大序列长度
            return_tensors: 返回张量格式 ('np' 或 'pt')
            
        Returns:
            tokenized结果字典
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        
        processed = self.tokenizer(
            texts,
            return_tensors=return_tensors,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_token_type_ids=False
        )
        
        return processed
    
    def process_tokenized_input(
        self,
        tokenized_inputs: List[List[int]],
        max_length: int = 512
    ) -> Dict[str, np.ndarray]:
        """
        处理已tokenized的输入
        
        Args:
            tokenized_inputs: token ID列表
            max_length: 最大序列长度
            
        Returns:
            处理后的输入字典
        """
        input_ids = []
        attention_masks = []
        
        for tokens in tokenized_inputs:
            truncated = tokens[:max_length]
            pad_length = max_length - len(truncated)
            padded = truncated + [self.pad_token_id] * pad_length
            input_ids.append(padded)
            attention_masks.append([1] * len(truncated) + [0] * pad_length)
        
        return {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "attention_mask": np.array(attention_masks, dtype=np.int64),
            "total_tokens": sum(len(t[:max_length]) for t in tokenized_inputs)
        }
    
    def inference(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray
    ) -> np.ndarray:
        """
        执行推理
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            
        Returns:
            embeddings数组
        """
        if self.session is None:
            raise RuntimeError("ONNX session not loaded")
        
        feed_dict = {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64)
        }
        
        outputs = self.session.run(None, feed_dict)
        
        # 提取embeddings并进行L2归一化
        embeddings = outputs[0][:, 0]  # 使用CLS token
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        if hasattr(self, 'tokenizer_executor'):
            self.tokenizer_executor.shutdown(wait=False)

