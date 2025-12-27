#!/usr/bin/env python3
"""
测试BGE Embedding服务

使用方法:
    python scripts/test_embedding.py --url <服务地址>
    
示例:
    python scripts/test_embedding.py --url http://localhost:8080
"""
import argparse
import requests
import json
import numpy as np
from typing import List, Dict


def test_single_text(url: str, text: str):
    """测试单个文本"""
    print(f"\n{'='*60}")
    print(f"测试单个文本: {text[:50]}...")
    print(f"{'='*60}")
    
    payload = {
        "input": text,
        "model": "bge-large-zh-v1.5"
    }
    
    try:
        response = requests.post(f"{url}/v1/embeddings", json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        embedding = result["data"][0]["embedding"]
        
        print(f" 请求成功")
        print(f" Embedding维度: {len(embedding)}")
        print(f" 前5维: {embedding[:5]}")
        print(f" L2范数: {np.linalg.norm(embedding):.6f}")
        print(f" 数值范围: [{min(embedding):.6f}, {max(embedding):.6f}]")
        print(f" 处理时间: {result['usage'].get('processing_time', 'N/A')}s")
        print(f" Token数: {result['usage']['total_tokens']}")
        
        # 检查归一化
        norm = np.linalg.norm(embedding)
        if abs(norm - 1.0) < 0.01:
            print(" L2归一化正确")
        else:
            print(f" L2归一化异常: {norm}")
        
        return True
    except Exception as e:
        print(f" 请求失败: {e}")
        return False


def test_batch_texts(url: str, texts: List[str]):
    """测试批量文本"""
    print(f"\n{'='*60}")
    print(f"测试批量文本: {len(texts)}个文本")
    print(f"{'='*60}")
    
    payload = {
        "input": texts,
        "model": "bge-large-zh-v1.5"
    }
    
    try:
        response = requests.post(f"{url}/v1/embeddings", json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        embeddings = [item["embedding"] for item in result["data"]]
        
        print(f" 请求成功")
        print(f" 返回embeddings数量: {len(embeddings)}")
        print(f" 每个embedding维度: {len(embeddings[0])}")
        print(f" 处理时间: {result['usage'].get('processing_time', 'N/A')}s")
        print(f" 总Token数: {result['usage']['total_tokens']}")
        
        # 检查所有embeddings的归一化
        norms = [np.linalg.norm(emb) for emb in embeddings]
        all_normalized = all(abs(norm - 1.0) < 0.01 for norm in norms)
        if all_normalized:
            print(" 所有embeddings L2归一化正确")
        else:
            print(f" 部分embeddings L2归一化异常")
            print(f"   范数范围: [{min(norms):.6f}, {max(norms):.6f}]")
        
        return True
    except Exception as e:
        print(f" 请求失败: {e}")
        return False


def test_base64_encoding(url: str, text: str):
    """测试base64编码格式"""
    print(f"\n{'='*60}")
    print(f"测试base64编码格式")
    print(f"{'='*60}")
    
    payload = {
        "input": text,
        "model": "bge-large-zh-v1.5",
        "encoding_format": "base64"
    }
    
    try:
        response = requests.post(f"{url}/v1/embeddings", json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        embedding_b64 = result["data"][0]["embedding"]
        
        print(f" 请求成功")
        print(f" Base64编码长度: {len(embedding_b64)}")
        print(f" Base64前50字符: {embedding_b64[:50]}...")
        
        # 解码验证
        import base64
        decoded = base64.b64decode(embedding_b64)
        import struct
        embedding = struct.unpack(f'{len(decoded)//4}f', decoded)
        print(f" 解码后维度: {len(embedding)}")
        print(f" 解码后前5维: {embedding[:5]}")
        
        return True
    except Exception as e:
        print(f" 请求失败: {e}")
        return False


def test_health(url: str):
    """测试健康检查"""
    print(f"\n{'='*60}")
    print(f"测试健康检查")
    print(f"{'='*60}")
    
    try:
        response = requests.get(f"{url}/health", timeout=5)
        response.raise_for_status()
        
        result = response.json()
        print(f" 健康检查成功")
        print(f" 状态: {result['status']}")
        print(f" 模型: {result['model']}")
        print(f" 版本: {result['model_version']}")
        print(f" Providers: {result['providers']}")
        print(f" 环境: {result['environment']}")
        
        return True
    except Exception as e:
        print(f" 健康检查失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="测试BGE Embedding服务")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080",
        help="服务地址（默认: http://localhost:8080）"
    )
    
    args = parser.parse_args()
    
    # 测试数据
    test_texts = [
        "介绍一下北京",
        "什么是人工智能",
        "机器学习的基本原理",
        "深度学习在自然语言处理中的应用",
        "BERT模型的工作原理"
    ]
    
    print("=" * 60)
    print("BGE Embedding服务测试")
    print("=" * 60)
    print(f"服务地址: {args.url}")
    
    results = []
    
    # 1. 健康检查
    results.append(("健康检查", test_health(args.url)))
    
    # 2. 单个文本测试
    results.append(("单个文本", test_single_text(args.url, test_texts[0])))
    
    # 3. 批量文本测试
    results.append(("批量文本", test_batch_texts(args.url, test_texts)))
    
    # 4. Base64编码测试
    results.append(("Base64编码", test_base64_encoding(args.url, test_texts[0])))
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("测试结果汇总")
    print(f"{'='*60}")
    for name, success in results:
        status = " 通过" if success else " 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    print(f"\n{'='*60}")
    if all_passed:
        print(" 所有测试通过!")
    else:
        print(" 部分测试失败")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

