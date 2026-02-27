#!/usr/bin/env python3
"""
Download RefCOCO dataset from HuggingFace and convert to Qwen-VL training format.

Usage:
    conda activate torch2.8
    cd qwen-vl-finetune
    python tools/prepare_refcoco.py \
        --output_dir ./data/refcoco_grounding \
        --num_train 2000 \
        --num_val 500 \
        --num_test 500
"""

import argparse
import json
import math
import os
import random
from pathlib import Path


def smart_resize(
    height: int, width: int, factor: int = 28,
    min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Qwen2.5-VL / Qwen3-VL image resize function.

    Rescales image so that:
    1. Both dimensions are divisible by 'factor'.
    2. Total pixels within [min_pixels, max_pixels].
    3. Aspect ratio maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def convert_bbox_to_resized(bbox, orig_h, orig_w, factor=28, min_pixels=56*56, max_pixels=50176):
    """Convert bbox from original image coords to resized image coords (Qwen3-VL format).

    Qwen3-VL uses absolute pixel coordinates of the resized image.
    """
    new_h, new_w = smart_resize(orig_h, orig_w, factor, min_pixels, max_pixels)
    scale_w = new_w / orig_w
    scale_h = new_h / orig_h

    x1, y1, x2, y2 = bbox
    x1_new = round(x1 * scale_w)
    y1_new = round(y1 * scale_h)
    x2_new = round(x2 * scale_w)
    y2_new = round(y2 * scale_h)

    # Clamp to resized image bounds
    x1_new = max(0, min(x1_new, new_w - 1))
    y1_new = max(0, min(y1_new, new_h - 1))
    x2_new = max(0, min(x2_new, new_w - 1))
    y2_new = max(0, min(y2_new, new_h - 1))

    return [x1_new, y1_new, x2_new, y2_new], new_h, new_w


def download_refcoco():
    """Download RefCOCO dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install datasets: pip install datasets"
        )

    print("Downloading RefCOCO from HuggingFace...")
    print("This may take a while on first run.")
    dataset = load_dataset("lmms-lab/RefCOCO")
    return dataset


def convert_sample(sample, image_dir: str, idx: int,
                   max_pixels: int = 50176, min_pixels: int = 784,
                   debug: bool = False) -> dict:
    """Convert a single RefCOCO sample to Qwen-VL grounding format.

    lmms-lab/RefCOCO HuggingFace fields:
        - image: PIL Image
        - question: str (the referring expression / query)
        - answer: str (answer text, may not be needed)
        - bbox: list [x, y, w, h] (COCO format)
        - file_name: str (original COCO filename)
    """
    if debug:
        print(f"  [DEBUG] Sample keys: {list(sample.keys())}")
        for k, v in sample.items():
            if k == "image":
                print(f"    {k}: PIL Image, size={v.size if hasattr(v, 'size') else '?'}")
            elif k == "segmentation":
                print(f"    {k}: (truncated)")
            else:
                print(f"    {k}: {repr(v)}")

    # In lmms-lab/RefCOCO, the 'question' field is a generic prompt template,
    # while the actual referring expressions are in the 'answer' field.
    # e.g. answer = ['second elephant from the left', '2nd ele from left']
    query = None

    # Try 'answer' first (RefCOCO HuggingFace format)
    answer = sample.get("answer", None)
    if answer:
        if isinstance(answer, list) and len(answer) > 0:
            query = str(answer[0]).strip()
        elif isinstance(answer, str):
            query = answer.strip()

    # Fallback to other common field names
    if not query:
        for key in ["sentence", "query", "sentences_raw", "sentences"]:
            if key in sample and sample[key]:
                val = sample[key]
                if isinstance(val, list):
                    query = val[0] if val else None
                else:
                    query = str(val).strip()
                if query:
                    break

    if not query:
        if debug:
            print("  [DEBUG] No query found, skipping")
        return None

    # Extract and convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
    bbox = sample.get("bbox", None)
    if bbox is None:
        if debug:
            print("  [DEBUG] No bbox found, skipping")
        return None

    # Handle different bbox formats
    if isinstance(bbox, dict):
        # Some datasets use {"x": ..., "y": ..., "w": ..., "h": ...}
        x = bbox.get("x", bbox.get("x1", 0))
        y = bbox.get("y", bbox.get("y1", 0))
        w = bbox.get("w", bbox.get("width", 0))
        h = bbox.get("h", bbox.get("height", 0))
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x, y, w, h = bbox
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    else:
        if debug:
            print(f"  [DEBUG] Unexpected bbox format: {type(bbox)} = {bbox}")
        return None

    # Validate bbox
    if x2 <= x1 or y2 <= y1:
        if debug:
            print(f"  [DEBUG] Invalid bbox: [{x1},{y1},{x2},{y2}]")
        return None

    # Save image to disk
    image = sample.get("image", None)
    if image is None:
        if debug:
            print("  [DEBUG] No image found, skipping")
        return None

    # Get image dimensions for coordinate conversion
    img_w, img_h = image.size

    image_filename = f"refcoco_{idx:06d}.jpg"
    image_path = os.path.join(image_dir, image_filename)

    if not os.path.exists(image_path):
        # Convert to RGB if needed (some images may be RGBA or palette mode)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(image_path, "JPEG")

    # Convert bbox to Qwen3-VL format: absolute coordinates in resized image space
    # Qwen3-VL resizes images via smart_resize(), bbox must match the resized dimensions
    resized_bbox, new_h, new_w = convert_bbox_to_resized(
        [x1, y1, x2, y2], img_h, img_w,
        min_pixels=min_pixels, max_pixels=max_pixels
    )

    if debug:
        print(f"  [DEBUG] Original bbox: [{x1},{y1},{x2},{y2}], Image: {img_w}x{img_h}")
        print(f"  [DEBUG] Resized image: {new_w}x{new_h}, Resized bbox: {resized_bbox}")

    # Build Qwen-VL conversation format
    result = {
        "image": image_filename,
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\nLocate \"{query}\" in this image and output the bbox coordinates in JSON format."
            },
            {
                "from": "gpt",
                "value": json.dumps({"bbox_2d": resized_bbox})
            }
        ]
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Prepare RefCOCO data for Qwen-VL grounding fine-tuning"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data/refcoco_grounding",
        help="Output directory for converted data"
    )
    parser.add_argument("--num_train", type=int, default=2000, help="Number of training samples")
    parser.add_argument("--num_val", type=int, default=500, help="Number of validation samples")
    parser.add_argument("--num_test", type=int, default=500, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_pixels", type=int, default=50176,
                        help="Max pixels for smart_resize (must match training script)")
    parser.add_argument("--min_pixels", type=int, default=784,
                        help="Min pixels for smart_resize (must match training script)")
    args = parser.parse_args()

    print(f"Using max_pixels={args.max_pixels}, min_pixels={args.min_pixels}")
    print(f"(MUST match --max_pixels / --min_pixels in training script!)")

    random.seed(args.seed)

    # Create output directories
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Download dataset
    dataset = download_refcoco()

    # Inspect dataset structure
    print(f"Dataset splits: {list(dataset.keys())}")
    for split_name in dataset:
        print(f"  {split_name}: {len(dataset[split_name])} samples")
        if len(dataset[split_name]) > 0:
            print(f"  Columns: {dataset[split_name].column_names}")

    # Collect all samples from available splits
    all_samples = []
    for split_name in dataset:
        for sample in dataset[split_name]:
            all_samples.append(sample)

    print(f"\nTotal raw samples: {len(all_samples)}")

    # Shuffle and convert
    random.shuffle(all_samples)

    total_needed = args.num_train + args.num_val + args.num_test
    converted = []
    print(f"\nConverting samples (target: {total_needed})...")

    for i, sample in enumerate(all_samples):
        if len(converted) >= total_needed:
            break
        # Debug: print the first sample's structure to help diagnose issues
        debug = (i == 0)
        result = convert_sample(sample, str(image_dir), i,
                               max_pixels=args.max_pixels,
                               min_pixels=args.min_pixels, debug=debug)
        if result is not None:
            converted.append(result)
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1} raw samples, converted {len(converted)} so far...")

    if len(converted) < total_needed:
        print(f"WARNING: Only converted {len(converted)} / {total_needed} samples.")
        # Adjust split sizes proportionally
        ratio = len(converted) / total_needed
        args.num_train = int(args.num_train * ratio)
        args.num_val = int(args.num_val * ratio)
        args.num_test = len(converted) - args.num_train - args.num_val

    # Split into train / val / test
    train_data = converted[:args.num_train]
    val_data = converted[args.num_train:args.num_train + args.num_val]
    test_data = converted[args.num_train + args.num_val:args.num_train + args.num_val + args.num_test]

    # Save to JSONL files
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        filepath = output_dir / f"{name}.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(data)} samples to {filepath}")

    # Also save as single JSON for compatibility
    train_json_path = output_dir / "train.json"
    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"Saved train.json to {train_json_path}")

    print(f"\n=== Summary ===")
    print(f"Output directory: {output_dir}")
    print(f"Images directory: {image_dir}")
    print(f"Train: {len(train_data)} samples")
    print(f"Val:   {len(val_data)} samples")
    print(f"Test:  {len(test_data)} samples")
    print(f"\nNext steps:")
    print(f"  1. Register the dataset in qwenvl/data/__init__.py")
    print(f"  2. Run: bash scripts/sft_2b_lora_16g.sh")


if __name__ == "__main__":
    main()
