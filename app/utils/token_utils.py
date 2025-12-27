"""
Token处理工具
"""
import tiktoken
import logging
from typing import List

logger = logging.getLogger(__name__)

# 初始化tiktoken编码器
encoding = tiktoken.get_encoding("cl100k_base")


def decode_token_ids(token_ids: List[int]) -> str:
    """
    将token IDs解码为文本
    
    Args:
        token_ids: token ID列表
        
    Returns:
        解码后的文本
    """
    try:
        return encoding.decode(token_ids)
    except Exception as e:
        logger.warning(f"Token解码失败: {e}")
        raise ValueError(f"Failed to decode token IDs: {e}")

