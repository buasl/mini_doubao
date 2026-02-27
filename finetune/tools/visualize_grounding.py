#!/usr/bin/env python3
"""
Visualize grounding predictions: draw GT bbox (green) and predicted bbox (red) on images.

Usage:
    cd qwen-vl-finetune
    python tools/visualize_grounding.py \
        --eval_results ./output/eval_results.json \
        --image_dir ./data/refcoco_grounding/images \
        --output_dir ./output/visualizations \
        --num_images 20
"""

import argparse
import json
import math
import os

from PIL import Image, ImageDraw, ImageFont


def smart_resize(
    height: int, width: int, factor: int = 28,
    min_pixels: int = 56 * 56, max_pixels: int = 50176
):
    """Qwen3-VL image resize function (same as in prepare_refcoco.py)."""
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
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


def bbox_resized_to_original(bbox, orig_h, orig_w, min_pixels=784, max_pixels=50176):
    """Convert bbox from resized image coords back to original image coords."""
    if bbox is None:
        return None
    new_h, new_w = smart_resize(orig_h, orig_w, min_pixels=min_pixels, max_pixels=max_pixels)
    scale_w = orig_w / new_w
    scale_h = orig_h / new_h
    x1, y1, x2, y2 = bbox
    return [
        round(x1 * scale_w), round(y1 * scale_h),
        round(x2 * scale_w), round(y2 * scale_h)
    ]


def draw_bbox(draw, bbox, color, label=None, width=3):
    """Draw a bounding box on the image."""
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        # Draw label with white background for readability
        text_y = max(0, y1 - 18)
        text_bbox = draw.textbbox((x1, text_y), label)
        draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
            fill="white"
        )
        draw.text((x1, text_y), label, fill=color)


def main():
    parser = argparse.ArgumentParser(description="Visualize grounding results")
    parser.add_argument("--eval_results", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/visualizations")
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--max_pixels", type=int, default=50176,
                        help="Max pixels used in training (must match)")
    parser.add_argument("--min_pixels", type=int, default=784,
                        help="Min pixels used in training (must match)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.eval_results, "r") as f:
        data = json.load(f)

    details = data["details"][:args.num_images]

    for item in details:
        image_path = os.path.join(args.image_dir, item["image"])
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size
        draw = ImageDraw.Draw(img)

        # Convert bboxes from resized image coords back to original image coords
        gt_bbox = bbox_resized_to_original(
            item["gt_bbox"], orig_h, orig_w,
            min_pixels=args.min_pixels, max_pixels=args.max_pixels
        )
        pred_bbox = bbox_resized_to_original(
            item.get("pred_bbox"), orig_h, orig_w,
            min_pixels=args.min_pixels, max_pixels=args.max_pixels
        )

        # Draw GT bbox in green
        draw_bbox(draw, gt_bbox, color="lime", label="GT")

        # Draw predicted bbox in red
        draw_bbox(draw, pred_bbox, color="red", label="Pred")

        # Add query text and IoU score with white background
        query = item.get("query", "")[:60]
        iou = item.get("iou", 0)
        info_text = f"Q: {query}  |  IoU: {iou:.3f}"
        text_bbox = draw.textbbox((5, 5), info_text)
        draw.rectangle(
            [text_bbox[0] - 3, text_bbox[1] - 3, text_bbox[2] + 3, text_bbox[3] + 3],
            fill="white"
        )
        draw.text((5, 5), info_text, fill="black")

        # Save
        out_path = os.path.join(args.output_dir, f"vis_{item['index']:04d}.jpg")
        img.save(out_path, "JPEG", quality=90)

    print(f"Saved {len(details)} visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
