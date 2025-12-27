#!/usr/bin/env python3
"""
将BGE模型转换为ONNX格式

使用方法:
    python scripts/convert_to_onnx.py --model_path <模型路径> --output_path <输出路径>
    
示例:
    python scripts/convert_to_onnx.py --model_path D:\backup\models\BAAI\bge-large-zh-v1___5 --output_path bge-large-zh-v1.5.onnx
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModel
import onnx
import os
import sys
from pathlib import Path


def convert_to_onnx(
    model_path: str,
    output_path: str,
    opset_version: int = 13,
    max_length: int = 512
):
    """
    将BGE模型转换为ONNX格式
    
    Args:
        model_path: 原始模型路径（HuggingFace格式）
        output_path: 输出ONNX文件路径
        opset_version: ONNX opset版本
        max_length: 最大序列长度
    """
    print("=" * 80)
    print(" 开始转换BGE模型为ONNX格式")
    print("=" * 80)
    
    # 检查模型路径
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    print(f" 模型路径: {model_path}")
    print(f" 输出路径: {output_path}")
    print(f"  Opset版本: {opset_version}")
    print(f"  最大序列长度: {max_length}")
    print()
    
    # 1. 加载模型和分词器
    print(" 正在加载模型和分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        print(f" 模型加载成功")
        print(f"   词汇表大小: {tokenizer.vocab_size}")
        print(f"   模型类型: {type(model).__name__}")
    except Exception as e:
        print(f" 模型加载失败: {e}")
        raise
    
    # 2. 构造示例输入
    print("\n 构造示例输入...")
    text = "这是一个用于导出ONNX的示例句子。"
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True
    )
    print(f" 示例输入构造完成")
    print(f"   Input IDs shape: {inputs['input_ids'].shape}")
    print(f"   Attention Mask shape: {inputs['attention_mask'].shape}")
    
    # 3. 导出为ONNX
    print("\n 正在导出为ONNX格式...")
    try:
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "last_hidden_state": {0: "batch_size", 1: "sequence"},
                "pooler_output": {0: "batch_size"}
            },
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False
        )
        print(f" ONNX导出成功!")
    except Exception as e:
        print(f" ONNX导出失败: {e}")
        raise
    
    # 4. 验证ONNX模型
    print("\n 验证ONNX模型...")
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f" ONNX模型验证通过")
        
        # 显示模型信息
        print("\n 模型信息:")
        for input_tensor in onnx_model.graph.input:
            name = input_tensor.name
            shape = [
                dim.dim_value if dim.HasField('dim_value') else "?"
                for dim in input_tensor.type.tensor_type.shape.dim
            ]
            print(f"   输入: {name}, shape: {shape}")
        
        for output_tensor in onnx_model.graph.output:
            name = output_tensor.name
            shape = [
                dim.dim_value if dim.HasField('dim_value') else "?"
                for dim in output_tensor.type.tensor_type.shape.dim
            ]
            print(f"   输出: {name}, shape: {shape}")
        
        # 显示文件大小
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"\n 文件大小: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"  ONNX模型验证失败: {e}")
        print("   但文件已生成，请手动检查")
    
    print("\n" + "=" * 80)
    print(" 转换完成!")
    print(f" ONNX文件保存在: {os.path.abspath(output_path)}")
    print("\n 注意事项:")
    print("   1. 请确保tokenizer文件（tokenizer.json等）与ONNX文件一起使用")
    print("   2. Tokenizer路径需要配置在环境变量 BGE_TOKENIZER_PATH 中")
    print("   3. 或者修改 app/core/config.py 中的 TOKENIZER_PATH")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="将BGE模型转换为ONNX格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python scripts/convert_to_onnx.py \\
      --model_path D:\\backup\\models\\BAAI\\bge-large-zh-v1___5 \\
      --output_path bge-large-zh-v1.5.onnx
  
  # 指定opset版本和最大长度
  python scripts/convert_to_onnx.py \\
      --model_path D:\\backup\\models\\BAAI\\bge-large-zh-v1___5 \\
      --output_path bge-large-zh-v1.5.onnx \\
      --opset_version 13 \\
      --max_length 512
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="原始模型路径（HuggingFace格式）"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出ONNX文件路径"
    )
    
    parser.add_argument(
        "--opset_version",
        type=int,
        default=13,
        help="ONNX opset版本（默认: 13）"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="最大序列长度（默认: 512）"
    )
    
    args = parser.parse_args()
    
    try:
        convert_to_onnx(
            model_path=args.model_path,
            output_path=args.output_path,
            opset_version=args.opset_version,
            max_length=args.max_length
        )
    except Exception as e:
        print(f"\n 转换失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

