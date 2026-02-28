"""
性能基准测试脚本 —— 真实测量资源占用与推理延迟

功能：
1. 加载模型并记录加载时间、显存占用、系统内存
2. 对多种场景（纯文本、图片理解、PDF解析、视频理解）进行推理测试
3. 记录每个场景的首 token 延迟、总延迟、生成 token 数、吞吐量
4. 收集输入输出示例
5. 将所有数据保存为 JSON，供文档生成脚本使用

Usage:
    # Transformers 后端
    python scripts/benchmark.py --backend transformers

    # vLLM 后端
    conda activate vllm
    python scripts/benchmark.py --backend vllm
"""

import argparse
import json
import os
import sys
import time
import platform
import tempfile
from datetime import datetime

import torch
import psutil
from PIL import Image, ImageDraw, ImageFont

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_doubao_app import get_gpu_memory_nvidia_smi


# ─── 工具函数 ───────────────────────────────────────────────────────────────

def get_gpu_info() -> dict:
    """获取 GPU 信息和当前显存使用情况。"""
    if not torch.cuda.is_available():
        return {"available": False}
    info = {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
        "reserved_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 2),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated(0) / 1024**3, 2),
    }
    # 尝试获取 nvidia-smi 信息
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            info["nvidia_smi_used_mb"] = int(parts[0].strip())
            info["nvidia_smi_total_mb"] = int(parts[1].strip())
            info["nvidia_smi_gpu_util_pct"] = int(parts[2].strip())
    except Exception:
        pass
    return info


def get_system_info() -> dict:
    """获取系统信息。"""
    mem = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=True),
        "total_ram_gb": round(mem.total / 1024**3, 2),
        "available_ram_gb": round(mem.available / 1024**3, 2),
        "used_ram_gb": round(mem.used / 1024**3, 2),
        "ram_percent": mem.percent,
    }


def get_torch_info() -> dict:
    """获取 PyTorch / CUDA 版本信息。"""
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = str(torch.backends.cudnn.version())
    return info


def create_test_image(width=640, height=480, text="Test Image") -> str:
    """创建一张测试图片并返回临时文件路径。"""
    img = Image.new("RGB", (width, height), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    # 画一些简单图形
    draw.rectangle([50, 50, 200, 200], fill=(255, 100, 100))
    draw.ellipse([300, 100, 500, 300], fill=(100, 255, 100))
    draw.rectangle([150, 250, 450, 400], fill=(100, 100, 255))
    # 添加文字
    try:
        draw.text((width // 2 - 40, 20), text, fill=(0, 0, 0))
    except Exception:
        pass
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name)
    tmp.close()
    return tmp.name


def create_test_pdf(pages=3) -> str:
    """创建一个简单的测试 PDF 文件。"""
    try:
        import fitz as pymupdf
    except ImportError:
        return None

    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page(width=595, height=842)  # A4
        text = f"这是测试 PDF 的第 {i + 1} 页\n\n" \
               f"本页包含一些示例文本，用于测试模型的 PDF 文档理解能力。\n\n" \
               f"人工智能（AI）是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。\n" \
               f"深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的表示。"
        page.insert_text((72, 72), text, fontsize=14, fontname="china-s")
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    doc.save(tmp.name)
    doc.close()
    tmp.close()
    return tmp.name


def count_tokens(processor, text: str) -> int:
    """估算文本的 token 数。"""
    try:
        return len(processor.tokenizer.encode(text))
    except Exception:
        return len(text) // 2  # 粗略估算


# ─── 基准测试类 ──────────────────────────────────────────────────────────────

class BenchmarkRunner:
    def __init__(self, model_path: str, backend: str, max_new_tokens: int = 512,
                 gpu_memory_utilization: float = 0.9, max_model_len: int = 8192):
        self.model_path = model_path
        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.results = {
            "metadata": {
                "model_path": model_path,
                "backend": backend,
                "max_new_tokens": max_new_tokens,
                "timestamp": datetime.now().isoformat(),
            },
            "system_info": get_system_info(),
            "torch_info": get_torch_info(),
            "gpu_info_before_load": get_gpu_info(),
            "model_loading": {},
            "gpu_info_after_load": {},
            "benchmarks": [],
            "resource_summary": {},
        }

        # 加载模型
        print("=" * 60)
        print(f"[基准测试] 后端: {backend}, 模型: {model_path}")
        print("=" * 60)

        if torch.cuda.is_available() and backend != "vllm":
            torch.cuda.reset_peak_memory_stats()

        mem_before = psutil.Process().memory_info().rss / 1024**3
        smi_before = get_gpu_memory_nvidia_smi()
        gpu_before = torch.cuda.memory_allocated(0) / 1024**3 if (torch.cuda.is_available() and backend != "vllm") else smi_before["used_gb"]

        t0 = time.time()
        if backend == "vllm":
            from my_doubao_app import VllmDoubaoAssistant
            self.assistant = VllmDoubaoAssistant(
                model_path=model_path,
                max_new_tokens=max_new_tokens,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
            self.processor = None
        else:
            from my_doubao_app import DoubaoAssistant
            self.assistant = DoubaoAssistant(
                model_path=model_path,
                max_new_tokens=max_new_tokens,
            )
            self.processor = self.assistant.processor
        load_time = time.time() - t0

        mem_after = psutil.Process().memory_info().rss / 1024**3
        smi_after = get_gpu_memory_nvidia_smi()
        if backend == "vllm":
            gpu_after = smi_after["used_gb"]
        else:
            gpu_after = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0

        self.results["model_loading"] = {
            "load_time_seconds": round(load_time, 2),
            "system_memory_before_gb": round(mem_before, 2),
            "system_memory_after_gb": round(mem_after, 2),
            "system_memory_delta_gb": round(mem_after - mem_before, 2),
            "gpu_memory_before_gb": round(gpu_before, 2),
            "gpu_memory_after_gb": round(gpu_after, 2),
            "gpu_memory_delta_gb": round(gpu_after - gpu_before, 2),
        }
        self.results["gpu_info_after_load"] = get_gpu_info()

        print(f"\n✅ 模型加载完成，耗时 {load_time:.2f} 秒")
        print(f"   GPU 显存: {gpu_before:.2f} GB → {gpu_after:.2f} GB (增加 {gpu_after - gpu_before:.2f} GB)")
        print(f"   系统内存: {mem_before:.2f} GB → {mem_after:.2f} GB (增加 {mem_after - mem_before:.2f} GB)")

    def _run_single_benchmark(self, name: str, user_text: str,
                               media_path: str = None,
                               extra_images: list = None,
                               warmup: bool = False) -> dict:
        """运行单个基准测试场景。"""
        print(f"\n{'─' * 50}")
        print(f"{'[预热]' if warmup else '[测试]'} {name}")
        print(f"  输入: {user_text[:80]}{'...' if len(user_text) > 80 else ''}")
        if media_path:
            print(f"  媒体: {media_path}")
        if extra_images:
            print(f"  额外图片: {len(extra_images)} 张")

        # 清理 GPU 缓存
        is_vllm = self.backend == "vllm"
        if torch.cuda.is_available() and not is_vllm:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # nvidia-smi 基线（对 vLLM 有效）
        smi_before = get_gpu_memory_nvidia_smi()
        smi_peak = smi_before["used_gb"]

        gpu_before = torch.cuda.memory_allocated(0) / 1024**3 if (torch.cuda.is_available() and not is_vllm) else smi_before["used_gb"]
        mem_before = psutil.Process().memory_info().rss / 1024**3

        conversation = []
        answer = ""
        first_token_time = None
        token_count = 0

        t_start = time.time()

        try:
            token_stream, updated_conversation = self.assistant.chat_stream(
                conversation, user_text, media_path,
                temperature=0.7, top_p=0.9,
                extra_images=extra_images,
            )

            for token in token_stream:
                if token.startswith("\r"):
                    continue
                if first_token_time is None:
                    first_token_time = time.time()
                answer += token
                token_count += 1
                # vLLM: 定期采样 nvidia-smi 峰值
                if is_vllm and token_count % 10 == 0:
                    cur_smi = get_gpu_memory_nvidia_smi()
                    smi_peak = max(smi_peak, cur_smi["used_gb"])

        except Exception as exc:
            answer = f"[错误] {exc}"
            print(f"  ❌ 错误: {exc}")

        t_end = time.time()

        total_time = t_end - t_start
        ttft = (first_token_time - t_start) if first_token_time else total_time

        # 最终显存采样
        smi_after = get_gpu_memory_nvidia_smi()
        smi_peak = max(smi_peak, smi_after["used_gb"])

        if is_vllm:
            gpu_after_peak = smi_peak
            gpu_after = smi_after["used_gb"]
        else:
            gpu_after_peak = torch.cuda.max_memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
            gpu_after = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
        mem_after = psutil.Process().memory_info().rss / 1024**3

        # 估算输出 token 数
        output_tokens = 0
        if self.processor:
            output_tokens = count_tokens(self.processor, answer.strip())
        else:
            output_tokens = len(answer.strip()) // 2  # 粗略估算

        tokens_per_sec = output_tokens / total_time if total_time > 0 else 0

        result = {
            "name": name,
            "input_text": user_text,
            "has_media": media_path is not None,
            "has_extra_images": bool(extra_images),
            "output_text": answer.strip(),
            "output_tokens_est": output_tokens,
            "timing": {
                "total_seconds": round(total_time, 3),
                "time_to_first_token_seconds": round(ttft, 3),
                "tokens_per_second": round(tokens_per_sec, 2),
            },
            "memory": {
                "gpu_before_gb": round(gpu_before, 2),
                "gpu_peak_gb": round(gpu_after_peak, 2),
                "gpu_after_gb": round(gpu_after, 2),
                "gpu_inference_delta_gb": round(gpu_after_peak - gpu_before, 2),
                "system_memory_before_gb": round(mem_before, 2),
                "system_memory_after_gb": round(mem_after, 2),
            },
        }

        print(f"  ✅ 完成: {total_time:.2f}s (首token {ttft:.2f}s), "
              f"~{output_tokens} tokens, {tokens_per_sec:.1f} tok/s")
        print(f"  GPU 峰值: {gpu_after_peak:.2f} GB, 推理增量: {gpu_after_peak - gpu_before:.2f} GB")
        print(f"  回复: {answer.strip()[:120]}{'...' if len(answer.strip()) > 120 else ''}")

        return result

    def run_all_benchmarks(self):
        """运行所有基准测试场景。"""

        # ── 预热 ──
        print("\n" + "=" * 60)
        print("[阶段 1/2] 预热推理引擎...")
        print("=" * 60)
        self._run_single_benchmark(
            "预热", "你好", warmup=True
        )

        # ── 正式测试 ──
        print("\n" + "=" * 60)
        print("[阶段 2/2] 正式基准测试")
        print("=" * 60)

        # 场景 1: 纯文本短问答
        r1 = self._run_single_benchmark(
            "纯文本短问答",
            "你好，请用两三句话介绍一下你自己。"
        )
        self.results["benchmarks"].append(r1)

        # 场景 2: 纯文本长回答
        r2 = self._run_single_benchmark(
            "纯文本长回答",
            "请详细解释什么是深度学习，包括其基本原理、主要架构（如CNN、RNN、Transformer）、"
            "训练过程以及在计算机视觉和自然语言处理中的应用。"
        )
        self.results["benchmarks"].append(r2)

        # 场景 3: 图片理解
        test_img = create_test_image(640, 480, "Benchmark Test")
        r3 = self._run_single_benchmark(
            "图片理解",
            "请详细描述这张图片中的内容，包括颜色、形状和布局。",
            media_path=test_img,
        )
        self.results["benchmarks"].append(r3)
        os.unlink(test_img)

        # 场景 4: PDF 文档解析
        test_pdf = create_test_pdf(3)
        if test_pdf:
            try:
                from my_doubao_app import pdf_to_images
                pdf_pages = pdf_to_images(test_pdf, max_pages=3)
                r4 = self._run_single_benchmark(
                    "PDF文档解析(3页)",
                    "请总结这份文档的主要内容。",
                    extra_images=pdf_pages,
                )
                self.results["benchmarks"].append(r4)
                for p in pdf_pages:
                    os.unlink(p)
                os.unlink(test_pdf)
            except Exception as exc:
                print(f"  ⚠️ PDF 测试跳过: {exc}")
        else:
            print("  ⚠️ PDF 测试跳过: PyMuPDF 未安装")

        # 场景 5: 多轮对话（模拟两轮）
        r5 = self._run_single_benchmark(
            "多轮对话-第1轮",
            "中国的首都是哪里？"
        )
        self.results["benchmarks"].append(r5)

        # 场景 6: 数学/推理
        r6 = self._run_single_benchmark(
            "数学推理",
            "一个水池有两个进水管和一个出水管。进水管A每小时注入3吨水，进水管B每小时注入2吨水，"
            "出水管每小时排出1吨水。如果水池容量为40吨，从空池开始，需要多少小时才能注满？请列出计算过程。"
        )
        self.results["benchmarks"].append(r6)

        # ── 汇总 ──
        self._compute_summary()

    def _compute_summary(self):
        """计算资源占用汇总。"""
        benchmarks = self.results["benchmarks"]
        if not benchmarks:
            return

        gpu_peaks = [b["memory"]["gpu_peak_gb"] for b in benchmarks]
        total_times = [b["timing"]["total_seconds"] for b in benchmarks]
        ttfts = [b["timing"]["time_to_first_token_seconds"] for b in benchmarks]
        tps_list = [b["timing"]["tokens_per_second"] for b in benchmarks if b["timing"]["tokens_per_second"] > 0]

        self.results["resource_summary"] = {
            "model_load_gpu_gb": self.results["model_loading"]["gpu_memory_delta_gb"],
            "model_load_time_seconds": self.results["model_loading"]["load_time_seconds"],
            "model_load_system_memory_gb": self.results["model_loading"]["system_memory_delta_gb"],
            "inference_gpu_peak_max_gb": round(max(gpu_peaks), 2),
            "inference_gpu_peak_avg_gb": round(sum(gpu_peaks) / len(gpu_peaks), 2),
            "avg_total_time_seconds": round(sum(total_times) / len(total_times), 2),
            "avg_ttft_seconds": round(sum(ttfts) / len(ttfts), 2),
            "avg_tokens_per_second": round(sum(tps_list) / len(tps_list), 2) if tps_list else 0,
            "gpu_total_memory_gb": self.results["gpu_info_after_load"].get("total_memory_gb", "N/A"),
        }

        print("\n" + "=" * 60)
        print("[汇总] 资源占用与性能")
        print("=" * 60)
        s = self.results["resource_summary"]
        print(f"  模型加载显存:     {s['model_load_gpu_gb']:.2f} GB")
        print(f"  模型加载时间:     {s['model_load_time_seconds']:.2f} 秒")
        print(f"  推理峰值显存(最大): {s['inference_gpu_peak_max_gb']:.2f} GB")
        print(f"  推理峰值显存(平均): {s['inference_gpu_peak_avg_gb']:.2f} GB")
        print(f"  平均总延迟:       {s['avg_total_time_seconds']:.2f} 秒")
        print(f"  平均首token延迟:  {s['avg_ttft_seconds']:.2f} 秒")
        print(f"  平均吞吐量:       {s['avg_tokens_per_second']:.1f} tokens/s")

    def save_results(self, output_path: str):
        """保存结果到 JSON 文件。"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"\n📄 基准测试结果已保存到: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-VL 性能基准测试")
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--backend", type=str, default="transformers",
                        choices=["transformers", "vllm"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--output", type=str, default=None,
                        help="输出 JSON 文件路径 (默认: benchmark_results_{backend}.json)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output or f"benchmark_results_{args.backend}.json"

    runner = BenchmarkRunner(
        model_path=args.model_path,
        backend=args.backend,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    runner.run_all_benchmarks()
    runner.save_results(output_path)

    print("\n✅ 基准测试完成！")
    print(f"   结果文件: {output_path}")
    print(f"   接下来运行: python scripts/generate_report.py --input {output_path}")


if __name__ == "__main__":
    main()
