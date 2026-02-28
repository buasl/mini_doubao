"""
技术说明文档生成脚本

读取 benchmark.py 生成的 JSON 结果文件，自动生成 Markdown 格式的技术说明文档。

Usage:
    python scripts/generate_report.py --input benchmark_results_transformers.json
    python scripts/generate_report.py --input benchmark_results_transformers.json --output TECHNICAL_REPORT.md
"""

import argparse
import json
import os
from datetime import datetime


def load_results(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_report(data: dict) -> str:
    meta = data.get("metadata", {})
    sys_info = data.get("system_info", {})
    torch_info = data.get("torch_info", {})
    gpu_info = data.get("gpu_info_after_load", {})
    loading = data.get("model_loading", {})
    benchmarks = data.get("benchmarks", [])
    summary = data.get("resource_summary", {})

    backend = meta.get("backend", "unknown")
    model_path = meta.get("model_path", "unknown")
    timestamp = meta.get("timestamp", datetime.now().isoformat())

    lines = []

    # ── 标题 ──
    lines.append("# 📊 Qwen3-VL-2B-Instruct 技术说明文档")
    lines.append("")
    lines.append(f"> 基准测试时间: {timestamp}")
    lines.append(f"> 推理后端: **{backend}**")
    lines.append(f"> 模型路径: `{model_path}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 1. 硬件与软件环境 ──
    lines.append("## 1. 硬件与软件环境")
    lines.append("")
    lines.append("| 项目 | 配置 |")
    lines.append("|------|------|")
    lines.append(f"| **操作系统** | {sys_info.get('platform', 'N/A')} |")
    lines.append(f"| **Python** | {sys_info.get('python_version', 'N/A')} |")
    lines.append(f"| **PyTorch** | {torch_info.get('torch_version', 'N/A')} |")
    lines.append(f"| **CUDA** | {torch_info.get('cuda_version', 'N/A')} |")
    lines.append(f"| **cuDNN** | {torch_info.get('cudnn_version', 'N/A')} |")
    lines.append(f"| **GPU** | {gpu_info.get('device_name', 'N/A')} |")
    lines.append(f"| **GPU 显存** | {gpu_info.get('total_memory_gb', 'N/A')} GB |")
    lines.append(f"| **CPU 核心数** | {sys_info.get('cpu_count', 'N/A')} |")
    lines.append(f"| **系统内存** | {sys_info.get('total_ram_gb', 'N/A')} GB |")
    lines.append(f"| **推理后端** | {backend} |")
    lines.append(f"| **最大生成 token** | {meta.get('max_new_tokens', 'N/A')} |")
    lines.append("")

    # ── 2. 资源占用 ──
    lines.append("## 2. 资源占用（实测数据）")
    lines.append("")
    lines.append("### 2.1 模型加载")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| **加载耗时** | {loading.get('load_time_seconds', 'N/A')} 秒 |")
    lines.append(f"| **GPU 显存占用** | {loading.get('gpu_memory_after_gb', 'N/A')} GB |")
    lines.append(f"| **GPU 显存增量** | {loading.get('gpu_memory_delta_gb', 'N/A')} GB |")
    lines.append(f"| **系统内存增量** | {loading.get('system_memory_delta_gb', 'N/A')} GB |")
    lines.append("")

    lines.append("### 2.2 推理资源占用")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| **推理峰值显存（最大）** | {summary.get('inference_gpu_peak_max_gb', 'N/A')} GB |")
    lines.append(f"| **推理峰值显存（平均）** | {summary.get('inference_gpu_peak_avg_gb', 'N/A')} GB |")
    lines.append(f"| **GPU 总显存** | {summary.get('gpu_total_memory_gb', 'N/A')} GB |")
    lines.append("")

    # ── 3. 推理延迟 ──
    lines.append("## 3. 推理延迟（实测数据）")
    lines.append("")
    lines.append("### 3.1 汇总")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| **平均总延迟** | {summary.get('avg_total_time_seconds', 'N/A')} 秒 |")
    lines.append(f"| **平均首 token 延迟** | {summary.get('avg_ttft_seconds', 'N/A')} 秒 |")
    lines.append(f"| **平均吞吐量** | {summary.get('avg_tokens_per_second', 'N/A')} tokens/s |")
    lines.append("")

    lines.append("### 3.2 各场景详细数据")
    lines.append("")
    lines.append("| 场景 | 总耗时(s) | 首token延迟(s) | 生成tokens | 吞吐量(tok/s) | GPU峰值(GB) |")
    lines.append("|------|-----------|----------------|------------|---------------|-------------|")
    for b in benchmarks:
        t = b.get("timing", {})
        m = b.get("memory", {})
        lines.append(
            f"| {b['name']} "
            f"| {t.get('total_seconds', 'N/A')} "
            f"| {t.get('time_to_first_token_seconds', 'N/A')} "
            f"| ~{b.get('output_tokens_est', 'N/A')} "
            f"| {t.get('tokens_per_second', 'N/A')} "
            f"| {m.get('gpu_peak_gb', 'N/A')} |"
        )
    lines.append("")

    # ── 4. 输入输出示例 ──
    lines.append("## 4. 输入输出示例（实际模型输出）")
    lines.append("")

    for i, b in enumerate(benchmarks, 1):
        name = b.get("name", f"场景{i}")
        input_text = b.get("input_text", "")
        output_text = b.get("output_text", "")
        has_media = b.get("has_media", False)
        has_extra = b.get("has_extra_images", False)

        lines.append(f"### 示例 {i}: {name}")
        lines.append("")

        media_note = ""
        if has_media:
            media_note = " [附带图片/视频]"
        elif has_extra:
            media_note = " [附带图片]"

        lines.append(f"**用户输入{media_note}：**")
        lines.append("```")
        lines.append(input_text)
        lines.append("```")
        lines.append("")
        lines.append("**模型输出：**")
        lines.append("```")
        lines.append(output_text)
        lines.append("```")
        lines.append("")

        t = b.get("timing", {})
        lines.append(f"*耗时: {t.get('total_seconds', '?')}s | "
                      f"首token: {t.get('time_to_first_token_seconds', '?')}s | "
                      f"吞吐: {t.get('tokens_per_second', '?')} tok/s*")
        lines.append("")

    # ── 5. 已知限制 ──
    lines.append("## 5. 已知限制")
    lines.append("")
    lines.append("| 限制 | 说明 |")
    lines.append("|------|------|")
    lines.append("| **PDF 页数上限** | 默认仅处理前 5 页（`max_pages=5`），超长文档需手动截取 |")
    lines.append("| **视频长度** | 受显存和 `max_model_len` 限制，建议视频不超过 30 秒 |")
    lines.append("| **并发能力** | 单用户单请求，不支持多用户并发推理 |")

    if backend == "vllm":
        lines.append('| **vLLM 流式** | vLLM 后端为"伪流式"（整体生成后分块输出），非逐 token 流式 |')
        lines.append('| **vLLM 显存监控** | 修复了之前显示 0 的问题，现通过 nvidia-smi 报告实际使用 |')
    else:
        lines.append("| **Transformers 流式** | 支持逐 token 流式输出，体感延迟较低 |")

    lines.append("| **URL 图片** | 仅支持直链图片 URL，不支持需要登录或 JS 渲染的页面 |")
    lines.append("| **多轮上下文** | 历史消息全部拼接，超长对话可能触发 OOM 或被截断。已新增自动裁剪逻辑，超长时会删除最早的轮次并插入系统提示 |")
    lines.append("| **CPU 模式** | 仅 Transformers 后端支持 `--cpu-only`，速度极慢，仅供调试 |")
    lines.append("| **模型能力** | 2B 参数量为轻量版，复杂推理和长文本生成能力弱于更大模型 |")

    # 根据实测数据添加显存相关限制
    gpu_total = summary.get("gpu_total_memory_gb", 0)
    gpu_peak = summary.get("inference_gpu_peak_max_gb", 0)
    if gpu_total and gpu_peak:
        usage_pct = (gpu_peak / gpu_total) * 100
        lines.append(f"| **显存使用率** | 推理峰值占 GPU 总显存的 {usage_pct:.1f}%，"
                      f"长上下文或多图输入可能导致 OOM |")

    lines.append("")

    # ── 6. 性能优化建议 ──
    lines.append("## 6. 性能优化建议")
    lines.append("")
    lines.append("1. **显存不足时**：降低 `--max-model-len` 或 `--gpu-memory-utilization` 参数")
    lines.append("2. **提升吞吐量**：使用 vLLM 后端（`--backend vllm`），适合批量推理场景")
    lines.append("3. **降低延迟**：使用 Transformers 后端的流式输出，首 token 延迟更低")
    lines.append("4. **长文档处理**：减少 PDF 页数或降低图片分辨率以节省显存")
    lines.append("5. **多轮对话**：定期清空会话历史，避免上下文过长导致 OOM")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"*本文档由 `scripts/benchmark.py` 和 `scripts/generate_report.py` 自动生成*")
    lines.append(f"*测试时间: {timestamp}*")
    lines.append("")

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="从基准测试结果生成技术说明文档")
    parser.add_argument("--input", type=str, required=True,
                        help="benchmark.py 生成的 JSON 结果文件路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 Markdown 文件路径 (默认: TECHNICAL_REPORT.md)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output or "TECHNICAL_REPORT.md"

    print(f"📖 读取基准测试结果: {args.input}")
    data = load_results(args.input)

    print(f"📝 生成技术说明文档...")
    report = generate_report(data)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ 技术说明文档已生成: {output_path}")
    print(f"   文档大小: {len(report)} 字符, {report.count(chr(10))} 行")


if __name__ == "__main__":
    main()
