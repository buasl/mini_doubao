# 完成报告：Qwen3-VL-2B LoRA 微调 — 视觉定位任务

## 新建/修改的文件

```
qwen-vl-finetune/
├── tools/
│   ├── prepare_refcoco.py          ← 下载 RefCOCO 并转换格式
│   ├── evaluate_grounding.py       ← 评估 IoU / Acc@0.5
│   └── visualize_grounding.py      ← 可视化 bbox 对比
├── scripts/
│   └── sft_2b_lora_16g.sh          ← 单卡 16GB 训练配置
├── qwenvl/data/__init__.py          ← 注册了 refcoco_grounding 数据集
└── data/refcoco_grounding/          ← (运行 prepare_refcoco.py 后生成)
    ├── images/
    ├── train.jsonl    (2000 条)
    ├── val.jsonl      (500 条)
    └── test.jsonl     (500 条)
```

## 使用步骤

### Step 1：准备数据
```bash
conda activate torch2.8
cd qwen-vl-finetune
pip install datasets   # 如未安装
python tools/prepare_refcoco.py --num_train 2000 --num_val 500 --num_test 500
```

### Step 2：训练

**单卡 16GB：**
```bash
bash scripts/sft_2b_lora_16g.sh
```

### Step 3：评估

**Baseline（未微调）：**
```bash
python tools/evaluate_grounding.py \
    --model_path ../checkpoint/Qwen3-VL-2B-Instruct \
    --test_file ./data/refcoco_grounding/test.jsonl \
    --image_dir ./data/refcoco_grounding/images \
    --output_file ./output/eval_baseline.json \
    --max_samples 50
```

**微调后 (LoRA)：**
```bash
python tools/evaluate_grounding.py \
    --model_path ../checkpoint/Qwen3-VL-2B-Instruct \
    --lora_path ./output/grounding_lora_16g/checkpoint-1600 \
    --test_file ./data/refcoco_grounding/test.jsonl \
    --image_dir ./data/refcoco_grounding/images \
    --output_file ./output/eval_lora.json
```

### Step 4：可视化
```bash
python tools/visualize_grounding.py \
    --eval_results ./output/eval_lora.json \
    --image_dir ./data/refcoco_grounding/images \
    --output_dir ./output/visualizations \
    --num_images 20
```


