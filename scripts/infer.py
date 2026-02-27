"""简单命令行推理示例

Usage:
    python infer.py [--model-path PATH] [--text TEXT]
"""
import argparse
from my_doubao_app import DoubaoAssistant


def parse_args():
    parser = argparse.ArgumentParser(description="命令行推理示例")
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct",
                        help="模型目录或 HuggingFace id")
    parser.add_argument("--text", type=str, default="你好",
                        help="要发送给模型的文本")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assistant = DoubaoAssistant(model_path=args.model_path)
    conv = []
    ans, conv = assistant.chat(conv, args.text, None)
    print("模型回复:", ans)
