#!/usr/bin/env python3
"""
Evaluate visual grounding performance on test set.
Computes IoU and Acc@0.5 between predicted and ground-truth bboxes.

Usage:
    conda activate torch2.8
    cd qwen-vl-finetune
    python tools/evaluate_grounding.py \
        --model_path Qwen/Qwen3-VL-2B-Instruct \
        --lora_path ./output/grounding_lora_16g \
        --test_file ./data/refcoco_grounding/test.jsonl \
        --image_dir ./data/refcoco_grounding/images \
        --output_file ./output/eval_results.json
"""

import argparse
import json
import re
import os

import torch
from PIL import Image


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0
    return intersection / union


def parse_bbox_from_response(response: str):
    """Parse bbox_2d from model response string."""
    # Try JSON parsing first
    try:
        data = json.loads(response)
        if "bbox_2d" in data:
            return data["bbox_2d"]
    except json.JSONDecodeError:
        pass

    # Fallback: regex to find [x1, y1, x2, y2]
    pattern = r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]'
    match = re.search(pattern, response)
    if match:
        return [int(match.group(i)) for i in range(1, 5)]

    return None


def load_model(model_path, lora_path=None):
    """Load Qwen3-VL model with optional LoRA adapter."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print(f"Loading model from {model_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if lora_path and os.path.exists(lora_path):
        print(f"Loading LoRA adapter from {lora_path}...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        print("LoRA adapter merged successfully.")

    processor = AutoProcessor.from_pretrained(model_path)

    model.eval()
    return model, processor


def run_inference(model, processor, image_path, query):
    """Run inference on a single image with a query."""
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    # Decode only the generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:]
    response = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response


def main():
    parser = argparse.ArgumentParser(description="Evaluate visual grounding")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="./output/eval_results.json")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--max_pixels", type=int, default=50176,
                        help="Max pixels for image resize (must match training script)")
    parser.add_argument("--min_pixels", type=int, default=784,
                        help="Min pixels for image resize (must match training script)")
    args = parser.parse_args()

    # Load test data
    test_data = []
    with open(args.test_file, "r") as f:
        for line in f:
            test_data.append(json.loads(line))
    print(f"Loaded {len(test_data)} test samples")

    if args.max_samples:
        test_data = test_data[:args.max_samples]
        print(f"Evaluating on {len(test_data)} samples")

    # Load model
    model, processor = load_model(args.model_path, args.lora_path)

    # Override processor image settings to match training
    ip = processor.image_processor
    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = args.min_pixels
        ip.max_pixels = args.max_pixels
        print(f"Set image_processor min_pixels={args.min_pixels}, max_pixels={args.max_pixels}")
    else:
        print("WARNING: Could not set min_pixels/max_pixels on image_processor")

    # Evaluate
    results = []
    ious = []
    correct_05 = 0
    correct_075 = 0

    for i, sample in enumerate(test_data):
        image_path = os.path.join(args.image_dir, sample["image"])
        query = sample["conversations"][0]["value"].replace("<image>\n", "")
        gt_bbox = json.loads(sample["conversations"][1]["value"])["bbox_2d"]

        try:
            response = run_inference(model, processor, image_path, query)
            pred_bbox = parse_bbox_from_response(response)
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            response = ""
            pred_bbox = None

        if pred_bbox:
            iou = compute_iou(pred_bbox, gt_bbox)
        else:
            iou = 0.0

        ious.append(iou)
        if iou >= 0.5:
            correct_05 += 1
        if iou >= 0.75:
            correct_075 += 1

        results.append({
            "index": i,
            "image": sample["image"],
            "query": query,
            "gt_bbox": gt_bbox,
            "pred_bbox": pred_bbox,
            "response": response,
            "iou": iou,
        })

        if (i + 1) % 10 == 0:
            avg_iou = sum(ious) / len(ious)
            acc_05 = correct_05 / len(ious) * 100
            print(f"  [{i+1}/{len(test_data)}] Avg IoU: {avg_iou:.4f}, Acc@0.5: {acc_05:.1f}%")

    # Final metrics
    avg_iou = sum(ious) / len(ious) if ious else 0
    acc_05 = correct_05 / len(ious) * 100 if ious else 0
    acc_075 = correct_075 / len(ious) * 100 if ious else 0

    metrics = {
        "num_samples": len(ious),
        "avg_iou": round(avg_iou, 4),
        "acc_at_0.5": round(acc_05, 2),
        "acc_at_0.75": round(acc_075, 2),
    }

    print(f"\n{'='*50}")
    print(f"=== Evaluation Results ===")
    print(f"  Samples: {metrics['num_samples']}")
    print(f"  Average IoU: {metrics['avg_iou']}")
    print(f"  Acc@0.5:  {metrics['acc_at_0.5']}%")
    print(f"  Acc@0.75: {metrics['acc_at_0.75']}%")
    print(f"{'='*50}")

    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    output = {"metrics": metrics, "details": results}
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
