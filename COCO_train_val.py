import os
import json
import re
import argparse
from typing import List, Dict, Any

import torch
from torchvision.datasets import CocoCaptions
from tqdm import tqdm

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor as QwenProcessor,
    MllamaForConditionalGeneration,
    AutoProcessor as LlamaProcessor,
)

from pycocoevalcap.spice.spice import Spice


# =========================
# CLI / Config
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate COCO paragraphs with Qwen2-VL and judge with Llama 3.2 Vision."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to COCO images root (e.g., data/coco/train2017 or data/coco/val2017).",
    )
    parser.add_argument(
        "--ann",
        type=str,
        required=True,
        help="Path to COCO captions JSON file (e.g., captions_train2017.json or captions_val2017.json).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Name of split for logging/output naming (e.g., train, val).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=-1,
        help="Max number of images to process (<=0 means use full dataset).",
    )
    parser.add_argument(
        "--caps-per-img",
        type=int,
        default=5,
        help="Number of COCO captions to use as hints per image.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max Qwen generations per image before giving up.",
    )
    parser.add_argument(
        "--llama-threshold",
        type=int,
        default=5,
        help="Accept paragraph if Llama score is greater than this (0–10).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Directory to save JSON outputs.",
    )
    return parser.parse_args()


args = parse_args()

SPLIT_NAME = args.split
CAPS_PER_IMG = args.caps_per_img
MAX_ATTEMPTS = args.max_attempts
LLAMA_THRESHOLD = args.llama_threshold

OUT_RAW_PATH = os.path.join(
    args.out_dir, f"generated_coco_{SPLIT_NAME}_qwen_llama_raw.json"
)
OUT_FINAL_PATH = os.path.join(
    args.out_dir, f"generated_coco_{SPLIT_NAME}_qwen_llama_with_spice.json"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
print("Split name:", SPLIT_NAME)


# =========================
# Load COCO dataset
# =========================
dataset = CocoCaptions(root=args.root, annFile=args.ann, transform=None)
num_images = len(dataset)
print(f"Total {SPLIT_NAME} images:", num_images)

if args.max_images is None or args.max_images <= 0:
    num_to_process = num_images
else:
    num_to_process = min(args.max_images, num_images)

print("Processing images:", num_to_process)


def get_image_path_and_id(ds: CocoCaptions, idx: int):
    img_id = ds.ids[idx]
    img_info = ds.coco.loadImgs(img_id)[0]
    file_name = img_info["file_name"]
    img_path = os.path.join(ds.root, file_name)
    return img_path, img_id


# =========================
# Load Qwen2 VL (generator)
# =========================
qwen_model_name = "Qwen/Qwen2-VL-7B-Instruct"

qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    qwen_model_name,
    device_map="auto",
    dtype=torch.bfloat16,
)
qwen_processor = QwenProcessor.from_pretrained(qwen_model_name)

print("Loaded Qwen2 VL generator.")


# =========================
# Load Llama 3.2 Vision (judge)
# =========================
llama_model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

llama_model = MllamaForConditionalGeneration.from_pretrained(
    llama_model_name,
    device_map="auto",
    dtype=torch.bfloat16,
)
llama_processor = LlamaProcessor.from_pretrained(llama_model_name)

print("Loaded Llama 3.2 Vision judge.")


# =========================
# Helpers: parsing scores, truncation for SPICE
# =========================
def parse_score(text: str) -> int:
    """
    Extract the last integer between 0 and 10 from Llama output.
    Return -1 if parsing fails.
    """
    matches = re.findall(r"(\d+)", text)
    if not matches:
        return -1
    n = int(matches[-1])
    return max(0, min(10, n))


def truncate_for_spice(text: str, max_words: int = 60) -> str:
    """
    Truncate very long paragraphs for SPICE to avoid cache/path issues.
    If text is None or empty, return an empty string.
    """
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


# =========================
# Qwen generation helper
# =========================
def qwen_generate_paragraph(image, captions: List[str]) -> str:
    """
    Generate one paragraph from Qwen2 VL based on image plus up to CAPS_PER_IMG captions as hints.
    """
    hints = "\n".join(f"- {c}" for c in captions)

    user_text = (
        "You are an expert image captioning assistant.\n"
        "Based on the image and the following example captions, write one single, fluent English paragraph "
        "of three to six sentences that thoroughly and concretely describes the image.\n"
        "Do not mention that you were given example captions. Just describe the image naturally.\n\n"
        "Example captions:\n"
        f"{hints}\n\n"
        "Paragraph:"
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        }
    ]

    chat_text = qwen_processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = qwen_processor(
        text=[chat_text],
        images=[image],
        return_tensors="pt",
    ).to(qwen_model.device)

    input_ids = inputs["input_ids"]

    with torch.no_grad():
        output_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )

    generated_ids = output_ids[:, input_ids.shape[1]:]

    paragraph = qwen_processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    return paragraph


# =========================
# Llama judge helper
# =========================
def llama_score_image_paragraph(image, paragraph: str) -> int:
    """
    Ask Llama 3.2 Vision to score how well `paragraph` describes `image`.
    Returns an integer in [0, 10], or -1 if parsing fails.
    """

    judge_instructions = (
        "You are a strict evaluator of image descriptions.\n"
        "You will see an image and a paragraph that claims to describe it.\n"
        "Give a score from 0 to 10 based only on how accurate and faithful "
        "the paragraph is to the image.\n"
        "0 means completely wrong, 5 means partially correct with important errors or omissions, "
        "and 10 means very accurate and detailed.\n"
        "Respond with a single integer number only, no extra words.\n\n"
        "Paragraph:\n"
        f"{paragraph}\n\n"
        "Score (0 to 10):"
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": judge_instructions},
            ],
        }
    ]

    chat_text = llama_processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = llama_processor(
        text=[chat_text],
        images=[image],
        return_tensors="pt",
    ).to(llama_model.device)

    input_ids = inputs["input_ids"]

    with torch.no_grad():
        output_ids = llama_model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )

    gen_ids = output_ids[:, input_ids.shape[1]:]

    out_text = llama_processor.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    score = parse_score(out_text)
    return score


# =========================
# Main loop: generate with Qwen, judge with Llama
# =========================
results: List[Dict[str, Any]] = []

for idx in tqdm(
    range(num_to_process),
    desc=f"Generating and judging paragraphs on {SPLIT_NAME}2017",
):
    image, captions = dataset[idx]
    captions = list(captions)[:CAPS_PER_IMG]
    if len(captions) == 0:
        continue

    img_path, img_id = get_image_path_and_id(dataset, idx)

    best_paragraph = None
    best_score = -1

    for attempt in range(MAX_ATTEMPTS):
        try:
            paragraph = qwen_generate_paragraph(image, captions)
        except Exception as e:
            print(f"Warning: Qwen generation failed for idx {idx}, attempt {attempt}: {e}")
            continue

        if not paragraph or not paragraph.strip():
            # empty output, try again
            continue

        try:
            score = llama_score_image_paragraph(image, paragraph)
        except Exception as e:
            print(f"Warning: Llama scoring failed for idx {idx}, attempt {attempt}: {e}")
            continue

        if score > best_score:
            best_score = score
            best_paragraph = paragraph

        if score > LLAMA_THRESHOLD:
            break

    if best_paragraph is None:
        print(f"Warning: no valid paragraph generated for image idx {idx}, id {img_id}")
        # store it anyway, with paragraph None and score -1
        results.append(
            {
                "image_id": int(img_id),
                "image_path": img_path,
                "split": SPLIT_NAME,
                "captions": captions,
                "paragraph": None,
                "llama_score": int(best_score),
            }
        )
    else:
        results.append(
            {
                "image_id": int(img_id),
                "image_path": img_path,
                "split": SPLIT_NAME,
                "captions": captions,
                "paragraph": best_paragraph,
                "llama_score": int(best_score),
            }
        )

    if (idx + 1) % 200 == 0:
        print("\nPreview at idx", idx)
        print("Captions:")
        for c in captions:
            print("  ", c)
        print("Best paragraph (score", best_score, "):")
        print(best_paragraph)
        print("-" * 80)


# =========================
# Save raw JSON before SPICE
# =========================
os.makedirs(args.out_dir, exist_ok=True)
with open(OUT_RAW_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nSaved raw generation results (without SPICE) to {OUT_RAW_PATH}")

# Report how many missing paragraphs
num_missing = sum(1 for item in results if not item["paragraph"])
print(f"Number of images with missing or empty paragraph: {num_missing} out of {len(results)}")


# =========================
# SPICE evaluation (safe)
# =========================
print(f"\nComputing SPICE scores for generated paragraphs on {SPLIT_NAME}2017...")

gts: Dict[int, List[str]] = {}
res: Dict[int, List[str]] = {}
index_map: Dict[int, int] = {}  # map spice index -> index in results

spice_idx = 0
for i, item in enumerate(results):
    paragraph = item["paragraph"]
    if not paragraph or not paragraph.strip():
        # skip items with no paragraph
        continue

    gts[spice_idx] = item["captions"]
    para_for_spice = truncate_for_spice(paragraph)
    res[spice_idx] = [para_for_spice]
    index_map[spice_idx] = i
    spice_idx += 1

print(f"Number of items used for SPICE: {spice_idx}")

if spice_idx > 0:
    spice_scorer = Spice()
    global_spice, spice_per_image = spice_scorer.compute_score(gts, res)
    print(f"Global SPICE over {spice_idx} {SPLIT_NAME} images: {global_spice:.4f}")

    # Initialize spice field for all items
    for item in results:
        item["spice"] = None

    # Assign SPICE scores back using index_map
    for s_idx, r_idx in index_map.items():
        spice_dict = spice_per_image[s_idx]
        try:
            results[r_idx]["spice"] = float(spice_dict["All"]["f"])
        except Exception as e:
            print(f"Warning: could not parse SPICE for spice index {s_idx}, result index {r_idx}: {e}")
            results[r_idx]["spice"] = None
else:
    print("No items had valid paragraphs for SPICE. Skipping SPICE scoring.")


# =========================
# Save final JSON with SPICE
# =========================
with open(OUT_FINAL_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Saved final results with SPICE to {OUT_FINAL_PATH}")


# python COCO_train_val.py \
#   --root data/coco/validation/val2017 \
#   --ann data/coco/annotations/captions_val2017.json \
#   --split validation \
#   --out-dir Dataset \
#   --max-images -1
