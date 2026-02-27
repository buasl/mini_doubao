# 小豆包（Qwen3-VL）部署与提交说明

> 说明：本项目为独立实现，代码文件为 `my_doubao_app.py`，未复用仓库原始 `web_demo_mm.py` 的实现逻辑。

## 1. 功能覆盖

- [x] 文本输入聊天
- [x] 图片输入问答
- [x] 视频输入问答
- [x] 多轮上下文对话（同一会话内保留历史）

## 2. 环境要求

- OS：Windows / Linux / macOS
- Python：3.10 ~ 3.12（建议 3.10 或 3.11）
- CUDA（可选）：有 NVIDIA GPU 时建议安装对应驱动

## 3. 硬件建议（参考）

- 最低可运行（CPU）：可运行但较慢
- 推荐（本地）：
  - GPU 显存 8GB+（Qwen3-VL-2B-Instruct）
  - 系统内存 16GB+
- 更高规格模型需要更大显存

## 4. 安装步骤

在项目根目录执行：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

python -m pip install -U pip
python -m pip install -r requirements_doubao.txt
```

## 5. 启动命令

### 5.1 本地模型路径启动（推荐）

```bash
python my_doubao_app.py --model-path ./Qwen3-VL-2B-Instruct --host 127.0.0.1 --port 7861 --inbrowser
```

### 5.2 Hugging Face 模型名启动

```bash
python my_doubao_app.py --model-path Qwen/Qwen3-VL-2B-Instruct --host 127.0.0.1 --port 7861 --inbrowser
```

### 5.3 纯 CPU 测试（慢）

```bash
python my_doubao_app.py --model-path ./Qwen3-VL-2B-Instruct --cpu-only
```

## 6. 使用方式

1. 在输入框输入文本问题并发送。
2. 可上传图片或视频，再配合文本提问。
3. 连续提问可保留上下文，形成多轮对话。
4. 点击“清空会话”可重置历史。

## 7. 提交材料建议

请在作业提交中包含：

1. **环境信息**
   - Python 版本
   - 关键依赖版本（`transformers`, `gradio`, `torch`, `qwen-vl-utils`）
2. **启动命令**
   - 你实际运行成功的命令
3. **硬件信息**
   - GPU 型号/显存、CPU、内存
4. **Demo 证据**
   - 截图（建议至少 3 张）：
     - 纯文本多轮对话
     - 图片问答
     - 视频问答
   - 或者提供全程录屏（从启动到三类能力演示）

## 8. 录屏建议（避免丢分）

- 录屏从命令行启动开始，展示模型加载成功。
- 依次演示：
  - 文本多轮（至少 2~3 轮）
  - 图片问答
  - 视频问答
- 每个环节都展示输入和输出。
- 录屏结尾展示代码文件名 `my_doubao_app.py` 与运行窗口，证明为自实现。

## 9. 常见问题

- `No module named gradio`：
  - 执行 `python -m pip install -r requirements_doubao.txt`
- 显存不足：
  - 切换 `Qwen3-VL-2B-Instruct`
  - 关闭占用 GPU 的程序
  - 或使用 `--cpu-only` 验证功能
- 上传视频报错：
  - 确认 `qwen-vl-utils==0.0.14` 已安装
