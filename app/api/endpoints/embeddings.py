"""
Embedding端点
"""
import asyncio
import time
import base64
import numpy as np
import logging
from typing import List
from fastapi import APIRouter, HTTPException
from app.models.schemas import EmbeddingRequest, EmbeddingResponse, EmbeddingData, EmbeddingUsage
from app.core.lifespan import get_embedding_model
from app.core.config import ServiceConfig

logger = logging.getLogger(__name__)
router = APIRouter()


async def process_batch(
    model,
    inputs: List[str],
    max_length: int = 512
) -> dict:
    """
    处理单个batch的推理
    
    Args:
        model: EmbeddingModel实例
        inputs: 输入文本列表
        max_length: 最大序列长度
        
    Returns:
        包含embeddings和total_tokens的字典
    """
    # Tokenize 操作：使用独立的 tokenizer_executor（CPU 操作，可以并发）
    processed = await asyncio.run_in_executor(
        model.tokenizer_executor,
        lambda: model.tokenize(inputs, max_length=max_length)
    )
    total_tokens = int(np.sum(processed["attention_mask"]))

    # 推理操作：使用推理 executor（GPU 操作，控制并发数为 1）
    embeddings = await asyncio.run_in_executor(
        model.executor,
        lambda: model.inference(
            processed["input_ids"],
            processed["attention_mask"]
        )
    )

    return {
        "embeddings": embeddings,
        "total_tokens": total_tokens
    }


@router.post("/v1/embeddings", response_model=EmbeddingResponse, tags=["embeddings"])
async def create_embeddings(request: EmbeddingRequest):
    """创建embeddings"""
    start_time = time.time()
    model = get_embedding_model()
    
    if model is None:
        raise HTTPException(500, "Model not initialized")
    
    # 处理输入：支持字符串或字符串列表
    if isinstance(request.input, str):
        inputs = [request.input]
    elif isinstance(request.input, list):
        inputs = request.input
    else:
        raise HTTPException(400, "Invalid input format")
    
    # 记录输入特征
    batch_size = len(inputs)
    input_lengths = [len(text) for text in inputs]  # 字符长度
    
    # 批量大小验证
    if len(inputs) > ServiceConfig.MAX_BATCH_SIZE_ALL:
        raise HTTPException(413, f"Batch size exceeds limit {ServiceConfig.MAX_BATCH_SIZE_ALL}")
    
    try:
        if batch_size > ServiceConfig.MAX_BATCH_SIZE:
            logger.info(f"请求batch_size({batch_size})超过限制({ServiceConfig.MAX_BATCH_SIZE})，进行拆分处理")
            
            # 拆分处理
            all_embeddings = []
            all_tokens = 0
            sub_batch_count = (batch_size + ServiceConfig.MAX_BATCH_SIZE - 1) // ServiceConfig.MAX_BATCH_SIZE
            
            for i in range(0, batch_size, ServiceConfig.MAX_BATCH_SIZE):
                end_idx = min(i + ServiceConfig.MAX_BATCH_SIZE, batch_size)
                batch_inputs = inputs[i:end_idx]
                sub_batch_num = i // ServiceConfig.MAX_BATCH_SIZE + 1
                
                logger.info(f"处理第{sub_batch_num}/{sub_batch_count}个子batch: {len(batch_inputs)}个文本")
                
                # 处理子batch
                batch_result = await process_batch(
                    model,
                    batch_inputs,
                    ServiceConfig.MAX_SEQUENCE_LENGTH
                )
                all_embeddings.extend(batch_result["embeddings"])
                all_tokens += batch_result["total_tokens"]
            
            embeddings = np.array(all_embeddings)
            total_tokens = all_tokens
            
        else:
            # 原有逻辑处理小batch
            batch_result = await process_batch(
                model,
                inputs,
                ServiceConfig.MAX_SEQUENCE_LENGTH
            )
            embeddings = batch_result["embeddings"]
            total_tokens = batch_result["total_tokens"]
    except Exception as e:
        logger.exception(e)
        raise HTTPException(500, f"Inference error: {str(e)}")
    
    # 处理编码格式
    emb_data = []
    for i, emb in enumerate(embeddings):
        if request.encoding_format == "base64":
            bytes_data = emb.astype(np.float32).tobytes()
            emb_value = base64.b64encode(bytes_data).decode("utf-8")
        else:
            emb_value = emb.tolist()
        
        emb_data.append(EmbeddingData(
            object="embedding",
            index=i,
            embedding=emb_value
        ))
    
    # 在返回前记录处理结果
    total_time = time.time() - start_time
    avg_length = np.mean(input_lengths) if input_lengths else 0
    logger.info(
        f"请求处理完成 | "
        f"耗时: {total_time:.3f}s | "
        f"批量: {batch_size} | "
        f"平均长度: {avg_length:.1f} | "
        f"总tokens: {total_tokens}"
    )
    
    return EmbeddingResponse(
        object="list",
        data=emb_data,
        model=request.model,
        usage=EmbeddingUsage(
            prompt_tokens=total_tokens,
            total_tokens=total_tokens
        )
    )

