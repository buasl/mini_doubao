import argparse
import copy
import os
import io
import tempfile
from threading import Thread
from typing import Any

import gradio as gr
import requests
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer

try:
    import fitz as pymupdf  # PyMuPDF

    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from qwen_vl_utils import process_vision_info

    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False

try:
    from vllm import LLM, SamplingParams

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".mpeg"}


def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def to_file_uri(path: str) -> str:
    return f"file://{os.path.abspath(path)}"


def pdf_to_images(pdf_path: str, max_pages: int = 5) -> list[str]:
    """将 PDF 前 max_pages 页转为临时 PNG 文件，返回文件路径列表。"""
    if not HAS_PYMUPDF:
        raise RuntimeError("需要安装 PyMuPDF 来处理 PDF 文件: pip install PyMuPDF")
    doc = pymupdf.open(pdf_path)
    paths: list[str] = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        pix = page.get_pixmap(dpi=200)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        pix.save(tmp.name)
        paths.append(tmp.name)
        tmp.close()
    doc.close()
    return paths


def url_to_image(url: str, timeout: int = 20) -> str:
    """从 URL 下载图片并保存为临时文件，返回文件路径。"""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"无效的 URL，请以 http:// 或 https:// 开头: {url}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "image/*,*/*;q=0.8",
    }
    resp = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if content_type and not any(t in content_type.lower() for t in ("image", "octet-stream")):
        raise ValueError(
            f"URL 返回的不是图片（Content-Type: {content_type}），请确认链接直接指向图片文件"
        )

    data = resp.content
    if len(data) < 100:
        raise ValueError("下载的内容太小，可能不是有效图片")

    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise ValueError(
            "无法解析为图片，请确认 URL 直接指向 jpg/png/webp 等图片文件，而非网页"
        )

    img = img.convert("RGB")
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name)
    tmp.close()
    return tmp.name


class DoubaoAssistant:
    def __init__(self, model_path: str, device_map: str = "auto", max_new_tokens: int = 512):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def _build_user_message(
        self,
        text: str,
        media_path: str | None,
        extra_images: list[str] | None = None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if media_path:
            if is_video(media_path):
                content.append({"type": "video", "video": to_file_uri(media_path)})
            else:
                content.append({"type": "image", "image": to_file_uri(media_path)})

        for img_path in (extra_images or []):
            content.append({"type": "image", "image": to_file_uri(img_path)})

        if text.strip():
            content.append({"type": "text", "text": text.strip()})

        if not content:
            raise ValueError("输入不能为空，请输入文本或上传图片/视频。")

        return {"role": "user", "content": content}

    def _prepare_model_inputs(self, messages: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        has_media = any(
            any(("image" in item) or ("video" in item) for item in message["content"])
            for message in messages
            if isinstance(message.get("content"), list)
        )

        if has_media:
            if not HAS_QWEN_VL_UTILS:
                raise RuntimeError(
                    "检测到图片/视频输入，但未安装 qwen-vl-utils。请先安装: pip install qwen-vl-utils==0.0.14"
                )

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=self.processor.image_processor.patch_size,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

            video_metadata = None
            if videos is not None:
                videos, video_metadata = zip(*videos)
                videos = list(videos)
                video_metadata = list(video_metadata)

            inputs = self.processor(
                text=[text],
                images=images,
                videos=videos,
                video_metadata=video_metadata,
                return_tensors="pt",
                padding=True,
                do_resize=False,
                **video_kwargs,
            )
        else:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True,
            )

        model_device = self.model.device
        return {k: v.to(model_device) for k, v in inputs.items()}

    def chat(self, conversation: list[dict[str, Any]], user_text: str, media_path: str | None) -> tuple[str, list[dict[str, Any]]]:
        new_conversation = copy.deepcopy(conversation)
        user_message = self._build_user_message(user_text, media_path)
        new_conversation.append(user_message)

        inputs = self._prepare_model_inputs(new_conversation)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        input_ids = inputs["input_ids"]
        generated_trimmed = generated_ids[:, input_ids.shape[1] :]
        answer = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        new_conversation.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            }
        )
        return answer, new_conversation

    def chat_stream(
        self,
        conversation: list[dict[str, Any]],
        user_text: str,
        media_path: str | None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        extra_images: list[str] | None = None,
    ) -> tuple[Any, list[dict[str, Any]]]:
        new_conversation = copy.deepcopy(conversation)
        user_message = self._build_user_message(user_text, media_path, extra_images=extra_images)
        new_conversation.append(user_message)

        inputs = self._prepare_model_inputs(new_conversation)
        streamer = TextIteratorStreamer(
            tokenizer=self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        generation_error: dict[str, Exception] = {}

        def _generate() -> None:
            try:
                gen_kwargs = dict(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    streamer=streamer,
                )
                if temperature > 1e-5:
                    gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
                else:
                    gen_kwargs.update(do_sample=False)
                self.model.generate(**gen_kwargs)
            except Exception as exc:  # noqa: BLE001
                generation_error["error"] = exc

        worker = Thread(target=_generate, daemon=True)
        worker.start()

        def _stream_tokens():
            for token_text in streamer:
                yield token_text
            worker.join()
            if "error" in generation_error:
                raise generation_error["error"]

        return _stream_tokens(), new_conversation


class VllmDoubaoAssistant:
    """使用 vLLM 后端进行推理的助手，接口与 DoubaoAssistant 兼容。"""

    def __init__(self, model_path: str, max_new_tokens: int = 512,
                 gpu_memory_utilization: float = 0.9, max_model_len: int = 8192):
        if not HAS_VLLM:
            raise RuntimeError("未安装 vllm，请先运行: pip install vllm")
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.llm = LLM(
            model=model_path,
            dtype="auto",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 10, "video": 2},
            allowed_local_media_path="/",
        )

    @staticmethod
    def _build_user_message(
        text: str,
        media_path: str | None,
        extra_images: list[str] | None = None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if media_path:
            if is_video(media_path):
                content.append({"type": "video_url", "video_url": {"url": to_file_uri(media_path)}})
            else:
                content.append({"type": "image_url", "image_url": {"url": to_file_uri(media_path)}})

        for img_path in (extra_images or []):
            content.append({"type": "image_url", "image_url": {"url": to_file_uri(img_path)}})

        if text.strip():
            content.append({"type": "text", "text": text.strip()})

        if not content:
            raise ValueError("输入不能为空，请输入文本或上传图片/视频。")

        return {"role": "user", "content": content}

    @staticmethod
    def _to_openai_messages(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """将内部 conversation 格式统一为 OpenAI 兼容格式供 vLLM 使用。"""
        result: list[dict[str, Any]] = []
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, str):
                result.append({"role": role, "content": content})
            elif isinstance(content, list):
                new_content: list[dict[str, Any]] = []
                for item in content:
                    if item.get("type") == "text":
                        new_content.append({"type": "text", "text": item["text"]})
                    elif item.get("type") == "image":
                        new_content.append({"type": "image_url", "image_url": {"url": item["image"]}})
                    elif item.get("type") == "image_url":
                        new_content.append(item)
                    elif item.get("type") == "video":
                        new_content.append({"type": "video_url", "video_url": {"url": item["video"]}})
                    elif item.get("type") == "video_url":
                        new_content.append(item)
                result.append({"role": role, "content": new_content})
            else:
                result.append({"role": role, "content": str(content)})
        return result

    def chat_stream(
        self,
        conversation: list[dict[str, Any]],
        user_text: str,
        media_path: str | None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        extra_images: list[str] | None = None,
    ) -> tuple[Any, list[dict[str, Any]]]:
        new_conversation = copy.deepcopy(conversation)
        user_message = self._build_user_message(user_text, media_path, extra_images=extra_images)
        new_conversation.append(user_message)

        openai_messages = self._to_openai_messages(new_conversation)

        sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=temperature if temperature > 1e-5 else 0.0,
            top_p=top_p,
        )

        import queue as _queue
        q: _queue.Queue = _queue.Queue()
        _DONE = object()

        def _generate() -> None:
            try:
                outputs = self.llm.chat(openai_messages, sampling_params=sampling_params)
                answer = outputs[0].outputs[0].text.strip()
                q.put(("ok", answer))
            except Exception as exc:  # noqa: BLE001
                q.put(("err", exc))
            finally:
                q.put((_DONE, None))

        worker = Thread(target=_generate, daemon=True)
        worker.start()

        def _stream_tokens():
            dot_count = 0
            while True:
                try:
                    tag, value = q.get(timeout=0.5)
                except _queue.Empty:
                    dot_count = (dot_count % 3) + 1
                    yield f"\r{'.' * dot_count}"
                    continue
                if tag is _DONE:
                    break
                if tag == "err":
                    raise value
                answer = value
                chunk_size = 8
                for i in range(0, len(answer), chunk_size):
                    yield answer[i : i + chunk_size]

        return _stream_tokens(), new_conversation


def build_demo(assistant: DoubaoAssistant | VllmDoubaoAssistant) -> gr.Blocks:
    def submit(
        text: str,
        media_file: str | None,
        pdf_file: str | None,
        image_url: str,
        chat_pairs: list[list[str]],
        conversation: list[dict[str, Any]],
        temperature: float,
        top_p: float,
    ):
        chat_pairs = chat_pairs or []
        conversation = conversation or []

        media_path = media_file if media_file else None
        user_text = text or ""
        extra_images: list[str] = []

        media_tag = ""
        if media_path:
            media_tag += f"\n📎 {os.path.basename(media_path)}"

        if pdf_file:
            try:
                pdf_pages = pdf_to_images(pdf_file)
                extra_images.extend(pdf_pages)
                media_tag += f"\n📄 PDF ({len(pdf_pages)} 页)"
            except Exception as exc:  # noqa: BLE001
                chat_pairs.append(["", f"[PDF 处理失败] {exc}"])
                yield "", None, None, "", chat_pairs, conversation
                return

        image_url_str = (image_url or "").strip()
        if image_url_str:
            try:
                url_img = url_to_image(image_url_str)
                extra_images.append(url_img)
                media_tag += f"\n🔗 {image_url_str[:60]}"
            except Exception as exc:  # noqa: BLE001
                chat_pairs.append(["", f"[URL 图片下载失败] {exc}"])
                yield "", None, None, "", chat_pairs, conversation
                return

        shown_user = (user_text.strip() if user_text.strip() else "[仅上传媒体]") + media_tag

        chat_pairs.append([shown_user, ""])
        yield "", None, None, "", chat_pairs, conversation

        try:
            token_stream, updated_conversation = assistant.chat_stream(
                conversation, user_text, media_path,
                temperature=temperature, top_p=top_p,
                extra_images=extra_images if extra_images else None,
            )
            answer = ""
            for token in token_stream:
                if token.startswith("\r"):
                    chat_pairs[-1][1] = answer + token.lstrip("\r")
                    yield "", None, None, "", chat_pairs, conversation
                    continue
                answer += token
                chat_pairs[-1][1] = answer
                yield "", None, None, "", chat_pairs, conversation

            answer = answer.strip()
            chat_pairs[-1][1] = answer
            updated_conversation.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                }
            )
            yield "", None, None, "", chat_pairs, updated_conversation
        except Exception as exc:  # noqa: BLE001
            answer = f"[错误] {exc}"
            chat_pairs[-1][1] = answer
            yield "", None, None, "", chat_pairs, conversation

    def clear_history():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [], []

    with gr.Blocks(title="小豆包（Qwen3-VL）") as demo:
        gr.Markdown("## 小豆包（Qwen3-VL 本地多模态聊天）")
        gr.Markdown("支持：文本 / 图片 / 视频 / PDF 文档 / 图片URL 输入，多轮上下文对话。")

        chatbot = gr.Chatbot(label="小豆包", height=520)
        conversation_state = gr.State([])

        with gr.Row():
            text_input = gr.Textbox(label="输入文本", lines=3, placeholder="请输入问题，或配合图片/视频/PDF提问")
            media_input = gr.File(label="上传图片或视频（可选）", file_types=["image", "video"], type="filepath")

        with gr.Row():
            pdf_input = gr.File(label="上传 PDF 文档（可选，最多识别前5页）", file_types=[".pdf"], type="filepath")
            url_input = gr.Textbox(label="图片 URL（可选）", lines=1, placeholder="粘贴网络图片链接，如 https://example.com/photo.jpg")

        with gr.Row():
            temperature_slider = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.05, label="Temperature", info="越高越随机，0 为贪心解码")
            top_p_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Top-P", info="核采样概率阈值")

        with gr.Row():
            send_btn = gr.Button("发送", variant="primary")
            clear_btn = gr.Button("清空会话")

        all_inputs = [text_input, media_input, pdf_input, url_input, chatbot, conversation_state, temperature_slider, top_p_slider]
        all_outputs = [text_input, media_input, pdf_input, url_input, chatbot, conversation_state]

        send_btn.click(submit, inputs=all_inputs, outputs=all_outputs)
        text_input.submit(submit, inputs=all_inputs, outputs=all_outputs)
        clear_btn.click(clear_history, outputs=[chatbot, conversation_state])

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local multimodal XiaoDoubao with Qwen3-VL")
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct", help="Local model path or HF model id")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--cpu-only", action="store_true", help="Run on CPU only (slow, transformers only)")
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--backend", type=str, default="transformers", choices=["transformers", "vllm"],
                        help="Inference backend: 'transformers' or 'vllm'")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="GPU memory utilization for vLLM (0.0~1.0)")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Max sequence length for vLLM KV cache (default 8192, reduce if OOM)")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--inbrowser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.backend == "vllm":
        print(f"[启动] 使用 vLLM 后端加载模型: {args.model_path}")
        assistant = VllmDoubaoAssistant(
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
    else:
        print(f"[启动] 使用 Transformers 后端加载模型: {args.model_path}")
        device_map = "cpu" if args.cpu_only else "auto"
        assistant = DoubaoAssistant(
            model_path=args.model_path,
            device_map=device_map,
            max_new_tokens=args.max_new_tokens,
        )
    app = build_demo(assistant)
    app.queue().launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=args.inbrowser,
        share=args.share,
    )


if __name__ == "__main__":
    main()
