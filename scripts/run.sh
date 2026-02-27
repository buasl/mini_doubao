#!/bin/bash
# 小豆包启动脚本
# 使用方式: bash scripts/run.sh [args]
# 示例:    bash scripts/run.sh --model-path ./Qwen3-VL-2B-Instruct --host 0.0.0.0 --port 7861

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
python my_doubao_app.py "$@"