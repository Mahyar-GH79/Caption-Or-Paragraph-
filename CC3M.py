"""
CC3M + Qwen2-VL + Llama 3.2 Vision data generation script.

For each valid CC3M image–caption pair:
  - Generate 5 positive captions (correct, diverse).
  - Generate 5 hard negative captions (similar style, but incorrect).
  - Generate one paragraph (3–6 sentences) using Qwen2-VL with image + caption.
  - Score the paragraph with Llama 3.2 Vision (0–10) in a feedback loop.

Results are written as JSONL: one JSON object per line.
You can control the target number of samples with MAX_SAMPLES.
"""

import os
import re
import json
from io import BytesIO
from typing import List, Dict, Any, Optional

import requests
from PIL import Image
from tqdm.auto import tqdm
from datasets import load_dataset

import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor as QwenProcessor,
    MllamaForConditionalGeneration,
    AutoProcessor as LlamaProcessor,
)

# =========================
# Config
# =========================

# Start with 5,000 for testing; later change to 500_000
MAX_SAMPLES       = 5000       # target number of VALID output samples

OUT_DIR           = "cc3m_qwen_llama_outputs"
OUT_JSONL_PATH    = os.path.join(
    OUT_DIR, f"cc3m_qwen_llama_{MAX_SAMPLES}.jsonl"
)

MAX_ATTEMPTS      = 3            # feedback attempts for paragraph
LLAMA_THRESHOLD   = 5            # accept paragraph if score > this
POS_CAPTIONS_N    = 5
NEG_CAPTIONS_N    = 5

CC3M_DATASET_NAME = "google-research-datasets/conceptual_captions"

os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# =========================
# Image download helper
# =========================

def download_image(url: str, timeout: float = 10.0) -> Optional[Image.Image]:
    """
    Download image from URL and return a RGB PIL.Image, or None on failure.
    """
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return img
    except Exception as e:
        # Comment this out if logs get too noisy during long runs.
        print(f"[skip] failed to download {url}: {e}")
        return None


# =========================
# Llama score parser
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


# =========================
# Model loading
# =========================

print("Loading Qwen2-VL and Llama 3.2 Vision models...")

qwen_model_name  = "Qwen/Qwen2-VL-7B-Instruct"
llama_model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    qwen_model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
qwen_processor = QwenProcessor.from_pretrained(qwen_model_name)

llama_model = MllamaForConditionalGeneration.from_pretrained(
    llama_model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
llama_processor = LlamaProcessor.from_pretrained(llama_model_name)

print("Models loaded.")


# =========================
# Qwen helpers
# =========================

def qwen_chat_with_image_and_caption(
    image: Image.Image,
    original_caption: str,
    user_text: str,
    max_new_tokens: int = 128,
) -> str:
    """
    Generic helper to call Qwen2-VL with both the image and its original caption
    plus extra instructions in user_text.
    """
    full_prompt = (
        "You are an expert image-language assistant.\n"
        "You will be given an image and its original caption from a dataset.\n"
        "Use both the visual content of the image and the caption information when following the instructions.\n\n"
        f"Original caption:\n{original_caption}\n\n"
        f"{user_text}"
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": full_prompt},
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
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )

    gen_ids = output_ids[:, input_ids.shape[1]:]
    out_text = qwen_processor.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()
    return out_text


def extract_numbered_list(text: str, expected_n: int) -> List[str]:
    """
    Parse a numbered list from text into a list of strings.
    Expected format:
        1. ...
        2) ...
        3 - ...
    Returns up to expected_n items.
    """
    lines = text.splitlines()
    items = []
    for line in lines:
        line = line.strip()
        if re.match(r"^\d+\s*[\.\)\-:]", line):
            item = re.sub(r"^\d+\s*[\.\)\-:]\s*", "", line).strip()
            if item:
                items.append(item)
    if not items and text:
        items = [text.strip()]
    return items[:expected_n]


def qwen_generate_positive_captions(
    image: Image.Image,
    original_caption: str,
    n: int = POS_CAPTIONS_N,
) -> List[str]:
    """
    Generate n positive captions using both the image and the original CC3M caption.
    """
    user_text = (
        "Using the given image and its original caption above, write five different short English captions.\n"
        "Each caption must be a correct description of the same image, but with different wording or focus.\n"
        "Each one should be a single sentence.\n"
        "Return them as a numbered list from 1 to 5."
    )
    raw = qwen_chat_with_image_and_caption(
        image,
        original_caption,
        user_text,
        max_new_tokens=160,
    )
    caps = extract_numbered_list(raw, n)
    return caps


def qwen_generate_hard_negative_captions(
    image: Image.Image,
    original_caption: str,
    n: int = NEG_CAPTIONS_N,
) -> List[str]:
    """
    Generate n hard negative captions using both the image and the original caption.
    They should be similar in style/length but clearly incorrect for this image.
    """
    user_text = (
        "The original caption above is a correct description of the image.\n"
        "Now you must create hard negative captions for image-text retrieval.\n"
        "Write five short English captions that are similar in style and length to the original caption, "
        "but that are NOT correct descriptions of this image.\n"
        "Each caption should change at least one important visual detail (objects, number of objects, colors, "
        "actions, relationships, or scene) so that a careful viewer can see it is wrong, but it should still "
        "sound plausible as a generic web image caption.\n"
        "Do NOT mention that the captions are negative or wrong.\n"
        "Return them as a numbered list from 1 to 5."
    )
    raw = qwen_chat_with_image_and_caption(
        image,
        original_caption,
        user_text,
        max_new_tokens=200,
    )
    caps = extract_numbered_list(raw, n)
    return caps


def qwen_generate_paragraph(
    image: Image.Image,
    original_caption: str,
) -> str:
    """
    Generate one paragraph (3–6 sentences) describing the image,
    using both the image and the original caption as context.
    """
    user_text = (
        "Using both the image and the original caption above, write a single, fluent English paragraph of "
        "three to six sentences that thoroughly and concretely describes the image.\n"
        "You may refine and expand on the original caption, but the paragraph must remain faithful to what "
        "is actually visible in the image.\n"
        "Describe the main objects, their attributes, relationships, and the overall scene.\n"
        "Do not mention that you were given a caption or that you are describing an image.\n\n"
        "Paragraph:"
    )
    paragraph = qwen_chat_with_image_and_caption(
        image,
        original_caption,
        user_text,
        max_new_tokens=200,
    )
    return paragraph.strip()


# =========================
# Llama scorer
# =========================

def llama_score_image_paragraph(
    image: Image.Image,
    paragraph: str,
) -> int:
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
# Main (robust streaming + skip bad samples)
# =========================

def main():
    print(f"Loading CC3M dataset (streaming): {CC3M_DATASET_NAME}")

    # Streaming load; do NOT use trust_remote_code (it is deprecated / unsupported)
    dataset = load_dataset(
        CC3M_DATASET_NAME,
        split="train",
        streaming=True,
    )

    iterator = iter(dataset)
    processed = 0   # number of raw CC3M examples we attempted
    written   = 0   # number of JSONL records actually written

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(OUT_JSONL_PATH, "w", encoding="utf-8") as f_out:
        pbar = tqdm(total=MAX_SAMPLES, desc="Generating CC3M samples")

        while written < MAX_SAMPLES:
            try:
                # Get next raw example
                try:
                    example = next(iterator)
                except StopIteration:
                    # Streaming iterator exhausted; re-open the stream
                    print("CC3M stream ended; restarting iterator.")
                    dataset = load_dataset(
                        CC3M_DATASET_NAME,
                        split="train",
                        streaming=True,
                    )
                    iterator = iter(dataset)
                    example = next(iterator)

                processed += 1

                caption = example.get("caption", None)
                url     = example.get("image_url", None)

                # Basic caption/url sanity checks
                if not caption or not isinstance(caption, str):
                    continue
                if not url or not isinstance(url, str):
                    continue

                image = download_image(url)
                if image is None:
                    continue

                # 1) Positive captions
                try:
                    pos_caps = qwen_generate_positive_captions(
                        image, caption, POS_CAPTIONS_N
                    )
                except Exception as e:
                    print(f"[warn] positive caption gen failed at processed={processed}: {e}")
                    pos_caps = []

                # 2) Hard negative captions
                try:
                    neg_caps = qwen_generate_hard_negative_captions(
                        image, caption, NEG_CAPTIONS_N
                    )
                except Exception as e:
                    print(f"[warn] negative caption gen failed at processed={processed}: {e}")
                    neg_caps = []

                # 3) Paragraph + Llama feedback loop
                best_paragraph = None
                best_score     = -1

                for attempt in range(MAX_ATTEMPTS):
                    try:
                        paragraph = qwen_generate_paragraph(image, caption)
                    except Exception as e:
                        print(f"[warn] Qwen paragraph failed at processed={processed}, attempt={attempt}: {e}")
                        continue

                    if not paragraph or not paragraph.strip():
                        continue

                    try:
                        score = llama_score_image_paragraph(image, paragraph)
                    except Exception as e:
                        print(f"[warn] Llama scoring failed at processed={processed}, attempt={attempt}: {e}")
                        continue

                    if score > best_score:
                        best_score     = score
                        best_paragraph = paragraph

                    if score > LLAMA_THRESHOLD:
                        break

                # If you want to enforce nonempty positives / paragraph, uncomment:
                # if not pos_caps or best_paragraph is None:
                #     continue

                record: Dict[str, Any] = {
                    "dataset": "CC3M",
                    "image_url": url,
                    "original_caption": caption,
                    "positive_captions": pos_caps,
                    "hard_negative_captions": neg_caps,
                    "paragraph": best_paragraph,
                    "llama_score": int(best_score),
                }

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                pbar.update(1)

                if written % 10 == 0:
                    print(f"\n[info] processed={processed}, written={written}")
                    print("Original caption:", caption)
                    print("Positive[0]:", pos_caps[0] if pos_caps else None)
                    print("Negative[0]:", neg_caps[0] if neg_caps else None)
                    print("Best score:", best_score)
                    snippet = (best_paragraph or "")[:200]
                    print("Paragraph snippet:", snippet, "...")
                    print("-" * 80)

            except KeyboardInterrupt:
                print("\n[info] Interrupted by user. Stopping early.")
                break
            except Exception as e:
                # Catch any unexpected error for this sample and move on
                print(f"[fatal-sample] unexpected error at processed={processed}: {e}")
                continue

    print(f"\nDone. Processed {processed} raw CC3M samples, wrote {written} JSONL lines to {OUT_JSONL_PATH}")


if __name__ == "__main__":
    main()









# """
# CC3M + Qwen2-VL + Llama 3.2 Vision data generation script.

# For each valid CC3M image–caption pair:
#   - Generate 5 positive captions (correct, diverse).
#   - Generate 5 hard negative captions (similar style, but incorrect).
#   - Generate one paragraph (3–6 sentences) using Qwen2-VL with image + caption.
#   - Score the paragraph with Llama 3.2 Vision (0–10) in a feedback loop.

# Results are written as JSONL: one JSON object per line.
# The script supports resuming from an existing JSONL file.
# """

# import os
# import re
# import json
# from io import BytesIO
# from typing import List, Dict, Any, Optional, Set

# import requests
# from PIL import Image
# from tqdm.auto import tqdm
# from datasets import load_dataset

# import torch
# from transformers import (
#     Qwen2VLForConditionalGeneration,
#     AutoProcessor as QwenProcessor,
#     MllamaForConditionalGeneration,
#     AutoProcessor as LlamaProcessor,
# )

# # =========================
# # Config
# # =========================

# # Start with 5_000 for testing. Later you can set 500_000.
# MAX_SAMPLES       = 5000

# OUT_DIR           = "cc3m_qwen_llama_outputs"
# OUT_JSONL_PATH    = os.path.join(
#     OUT_DIR, f"cc3m_qwen_llama_{MAX_SAMPLES}.jsonl"
# )

# MAX_ATTEMPTS      = 3            # feedback attempts for paragraph
# LLAMA_THRESHOLD   = 5            # accept paragraph if score > this
# POS_CAPTIONS_N    = 5
# NEG_CAPTIONS_N    = 5

# CC3M_DATASET_NAME = "google-research-datasets/conceptual_captions"

# os.makedirs(OUT_DIR, exist_ok=True)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using device:", device)


# # =========================
# # Image download helper
# # =========================

# def download_image(url: str, timeout: float = 10.0) -> Optional[Image.Image]:
#     """
#     Download image from URL and return a RGB PIL.Image, or None on failure.
#     """
#     if not url:
#         return None
#     try:
#         resp = requests.get(url, timeout=timeout)
#         resp.raise_for_status()
#         img = Image.open(BytesIO(resp.content)).convert("RGB")
#         return img
#     except Exception as e:
#         # For long runs you can silence this if too noisy.
#         print(f"[skip] failed to download {url}: {e}")
#         return None


# # =========================
# # Llama score parser
# # =========================

# def parse_score(text: str) -> int:
#     """
#     Extract the last integer between 0 and 10 from Llama output.
#     Return -1 if parsing fails.
#     """
#     matches = re.findall(r"(\d+)", text)
#     if not matches:
#         return -1
#     n = int(matches[-1])
#     return max(0, min(10, n))


# # =========================
# # Model loading
# # =========================

# print("Loading Qwen2-VL and Llama 3.2 Vision models...")

# qwen_model_name  = "Qwen/Qwen2-VL-7B-Instruct"
# llama_model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
#     qwen_model_name,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
# )
# qwen_processor = QwenProcessor.from_pretrained(qwen_model_name)

# llama_model = MllamaForConditionalGeneration.from_pretrained(
#     llama_model_name,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
# )
# llama_processor = LlamaProcessor.from_pretrained(llama_model_name)

# print("Models loaded.")


# # =========================
# # Qwen helpers
# # =========================

# def qwen_chat_with_image_and_caption(
#     image: Image.Image,
#     original_caption: str,
#     user_text: str,
#     max_new_tokens: int = 128,
# ) -> str:
#     """
#     Generic helper to call Qwen2-VL with both the image and its original caption
#     plus extra instructions in user_text.
#     """
#     full_prompt = (
#         "You are an expert image-language assistant.\n"
#         "You will be given an image and its original caption from a dataset.\n"
#         "Use both the visual content of the image and the caption information when following the instructions.\n\n"
#         f"Original caption:\n{original_caption}\n\n"
#         f"{user_text}"
#     )

#     conversation = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": image},
#                 {"type": "text", "text": full_prompt},
#             ],
#         }
#     ]

#     chat_text = qwen_processor.apply_chat_template(
#         conversation,
#         add_generation_prompt=True,
#         tokenize=False,
#     )

#     inputs = qwen_processor(
#         text=[chat_text],
#         images=[image],
#         return_tensors="pt",
#     ).to(qwen_model.device)

#     input_ids = inputs["input_ids"]

#     with torch.no_grad():
#         output_ids = qwen_model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=True,
#             top_p=0.9,
#             temperature=0.7,
#         )

#     gen_ids = output_ids[:, input_ids.shape[1]:]
#     out_text = qwen_processor.batch_decode(
#         gen_ids,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=True,
#     )[0].strip()
#     return out_text


# def extract_numbered_list(text: str, expected_n: int) -> List[str]:
#     """
#     Parse a numbered list from text into a list of strings.
#     Expected format:
#         1. ...
#         2) ...
#         3 - ...
#     Returns up to expected_n items.
#     """
#     lines = text.splitlines()
#     items = []
#     for line in lines:
#         line = line.strip()
#         if re.match(r"^\d+\s*[\.\)\-:]", line):
#             item = re.sub(r"^\d+\s*[\.\)\-:]\s*", "", line).strip()
#             if item:
#                 items.append(item)
#     if not items and text:
#         items = [text.strip()]
#     return items[:expected_n]


# def qwen_generate_positive_captions(
#     image: Image.Image,
#     original_caption: str,
#     n: int = POS_CAPTIONS_N,
# ) -> List[str]:
#     """
#     Generate n positive captions using both the image and the original CC3M caption.
#     """
#     user_text = (
#         "Using the given image and its original caption above, write five different short English captions.\n"
#         "Each caption must be a correct description of the same image, but with different wording or focus.\n"
#         "Each one should be a single sentence.\n"
#         "Return them as a numbered list from 1 to 5."
#     )
#     raw = qwen_chat_with_image_and_caption(
#         image,
#         original_caption,
#         user_text,
#         max_new_tokens=160,
#     )
#     caps = extract_numbered_list(raw, n)
#     return caps


# def qwen_generate_hard_negative_captions(
#     image: Image.Image,
#     original_caption: str,
#     n: int = NEG_CAPTIONS_N,
# ) -> List[str]:
#     """
#     Generate n hard negative captions using both the image and the original caption.
#     They should be similar in style or length but clearly incorrect for this image.
#     """
#     user_text = (
#         "The original caption above is a correct description of the image.\n"
#         "Now you must create hard negative captions for image-text retrieval.\n"
#         "Write five short English captions that are similar in style and length to the original caption, "
#         "but that are NOT correct descriptions of this image.\n"
#         "Each caption should change at least one important visual detail (objects, number of objects, colors, "
#         "actions, relationships, or scene) so that a careful viewer can see it is wrong, but it should still "
#         "sound plausible as a generic web image caption.\n"
#         "Do NOT mention that the captions are negative or wrong.\n"
#         "Return them as a numbered list from 1 to 5."
#     )
#     raw = qwen_chat_with_image_and_caption(
#         image,
#         original_caption,
#         user_text,
#         max_new_tokens=200,
#     )
#     caps = extract_numbered_list(raw, n)
#     return caps


# def qwen_generate_paragraph(
#     image: Image.Image,
#     original_caption: str,
# ) -> str:
#     """
#     Generate one paragraph (3–6 sentences) describing the image,
#     using both the image and the original caption as context.
#     """
#     user_text = (
#         "Using both the image and the original caption above, write a single, fluent English paragraph of "
#         "three to six sentences that thoroughly and concretely describes the image.\n"
#         "You may refine and expand on the original caption, but the paragraph must remain faithful to what "
#         "is actually visible in the image.\n"
#         "Describe the main objects, their attributes, relationships, and the overall scene.\n"
#         "Do not mention that you were given a caption or that you are describing an image.\n\n"
#         "Paragraph:"
#     )
#     paragraph = qwen_chat_with_image_and_caption(
#         image,
#         original_caption,
#         user_text,
#         max_new_tokens=200,
#     )
#     return paragraph.strip()


# # =========================
# # Llama scorer
# # =========================

# def llama_score_image_paragraph(
#     image: Image.Image,
#     paragraph: str,
# ) -> int:
#     """
#     Ask Llama 3.2 Vision to score how well `paragraph` describes `image`.
#     Returns an integer in [0, 10], or -1 if parsing fails.
#     """

#     judge_instructions = (
#         "You are a strict evaluator of image descriptions.\n"
#         "You will see an image and a paragraph that claims to describe it.\n"
#         "Give a score from 0 to 10 based only on how accurate and faithful "
#         "the paragraph is to the image.\n"
#         "0 means completely wrong, 5 means partially correct with important errors or omissions, "
#         "and 10 means very accurate and detailed.\n"
#         "Respond with a single integer number only, no extra words.\n\n"
#         "Paragraph:\n"
#         f"{paragraph}\n\n"
#         "Score (0 to 10):"
#     )

#     conversation = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": image},
#                 {"type": "text", "text": judge_instructions},
#             ],
#         }
#     ]

#     chat_text = llama_processor.apply_chat_template(
#         conversation,
#         add_generation_prompt=True,
#         tokenize=False,
#     )

#     inputs = llama_processor(
#         text=[chat_text],
#         images=[image],
#         return_tensors="pt",
#     ).to(llama_model.device)

#     input_ids = inputs["input_ids"]

#     with torch.no_grad():
#         output_ids = llama_model.generate(
#             **inputs,
#             max_new_tokens=16,
#             do_sample=False,
#         )

#     gen_ids = output_ids[:, input_ids.shape[1]:]
#     out_text = llama_processor.batch_decode(
#         gen_ids,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=True,
#     )[0].strip()

#     score = parse_score(out_text)
#     return score


# # =========================
# # Resume helper
# # =========================

# def load_existing_progress(path: str) -> (int, Set[str]):
#     """
#     If JSONL exists, load it and return:
#       - written: number of valid lines (records)
#       - used_urls: set of image_url values already processed

#     This lets us resume and skip previously used CC3M samples.
#     """
#     written = 0
#     used_urls: Set[str] = set()

#     if not os.path.exists(path):
#         return written, used_urls

#     print(f"[resume] Found existing file: {path}")
#     with open(path, "r", encoding="utf-8") as f_in:
#         for line in f_in:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 rec = json.loads(line)
#             except Exception:
#                 # Possible partial last line, just skip it.
#                 continue
#             written += 1
#             url = rec.get("image_url")
#             if isinstance(url, str) and url:
#                 used_urls.add(url)

#     print(f"[resume] Loaded {written} existing records, {len(used_urls)} unique URLs.")
#     return written, used_urls


# # =========================
# # Main (with resume)
# # =========================

# def main():
#     print(f"Loading CC3M dataset (streaming): {CC3M_DATASET_NAME}")

#     # Load existing progress if any
#     existing_written, used_urls = load_existing_progress(OUT_JSONL_PATH)

#     if existing_written >= MAX_SAMPLES:
#         print(f"[resume] Already have {existing_written} samples, which meets or exceeds MAX_SAMPLES={MAX_SAMPLES}. Nothing to do.")
#         return

#     # Streaming load; no trust_remote_code
#     dataset = load_dataset(
#         CC3M_DATASET_NAME,
#         split="train",
#         streaming=True,
#     )

#     iterator = iter(dataset)
#     processed = 0   # number of raw CC3M examples we attempted
#     written   = existing_written

#     # Append mode if resuming, write mode if fresh
#     mode = "a" if existing_written > 0 else "w"
#     os.makedirs(OUT_DIR, exist_ok=True)

#     with open(OUT_JSONL_PATH, mode, encoding="utf-8") as f_out:
#         pbar = tqdm(
#             total=MAX_SAMPLES,
#             initial=written,
#             desc="Generating CC3M samples",
#         )

#         while written < MAX_SAMPLES:
#             try:
#                 # Get next raw example from the CC3M stream
#                 try:
#                     example = next(iterator)
#                 except StopIteration:
#                     print("[info] CC3M stream ended; restarting iterator.")
#                     dataset = load_dataset(
#                         CC3M_DATASET_NAME,
#                         split="train",
#                         streaming=True,
#                     )
#                     iterator = iter(dataset)
#                     example = next(iterator)

#                 processed += 1

#                 caption = example.get("caption", None)
#                 url     = example.get("image_url", None)

#                 # Basic caption/url sanity checks
#                 if not caption or not isinstance(caption, str):
#                     continue
#                 if not url or not isinstance(url, str):
#                     continue

#                 # Skip if we already processed this URL in a previous run
#                 if url in used_urls:
#                     continue

#                 image = download_image(url)
#                 if image is None:
#                     continue

#                 # 1) Positive captions
#                 try:
#                     pos_caps = qwen_generate_positive_captions(
#                         image, caption, POS_CAPTIONS_N
#                     )
#                 except Exception as e:
#                     print(f"[warn] positive caption gen failed at processed={processed}: {e}")
#                     pos_caps = []

#                 # 2) Hard negative captions
#                 try:
#                     neg_caps = qwen_generate_hard_negative_captions(
#                         image, caption, NEG_CAPTIONS_N
#                     )
#                 except Exception as e:
#                     print(f"[warn] negative caption gen failed at processed={processed}: {e}")
#                     neg_caps = []

#                 # 3) Paragraph + Llama feedback loop
#                 best_paragraph = None
#                 best_score     = -1

#                 for attempt in range(MAX_ATTEMPTS):
#                     try:
#                         paragraph = qwen_generate_paragraph(image, caption)
#                     except Exception as e:
#                         print(f"[warn] Qwen paragraph failed at processed={processed}, attempt={attempt}: {e}")
#                         continue

#                     if not paragraph or not paragraph.strip():
#                         continue

#                     try:
#                         score = llama_score_image_paragraph(image, paragraph)
#                     except Exception as e:
#                         print(f"[warn] Llama scoring failed at processed={processed}, attempt={attempt}: {e}")
#                         continue

#                     if score > best_score:
#                         best_score     = score
#                         best_paragraph = paragraph

#                     if score > LLAMA_THRESHOLD:
#                         break

#                 record: Dict[str, Any] = {
#                     "dataset": "CC3M",
#                     "image_url": url,
#                     "original_caption": caption,
#                     "positive_captions": pos_caps,
#                     "hard_negative_captions": neg_caps,
#                     "paragraph": best_paragraph,
#                     "llama_score": int(best_score),
#                 }

#                 f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
#                 # Optional: flush occasionally for safety
#                 # if (written + 1) % 100 == 0:
#                 #     f_out.flush()
#                 #     os.fsync(f_out.fileno())

#                 written += 1
#                 used_urls.add(url)
#                 pbar.update(1)

#                 if written % 10 == 0:
#                     print(f"\n[info] processed={processed}, written={written}")
#                     print("Original caption:", caption)
#                     print("Positive[0]:", pos_caps[0] if pos_caps else None)
#                     print("Negative[0]:", neg_caps[0] if neg_caps else None)
#                     print("Best score:", best_score)
#                     snippet = (best_paragraph or "")[:200]
#                     print("Paragraph snippet:", snippet, "...")
#                     print("-" * 80)

#             except KeyboardInterrupt:
#                 print("\n[info] Interrupted by user. Stopping early.")
#                 break
#             except Exception as e:
#                 # Catch any unexpected error for this sample and move on
#                 print(f"[fatal-sample] unexpected error at processed={processed}: {e}")
#                 continue

#     print(f"\nDone. Processed {processed} raw CC3M samples, wrote {written} JSONL lines to {OUT_JSONL_PATH}")


# if __name__ == "__main__":
#     main()
