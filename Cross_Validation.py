import os
import json
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.datasets import CocoCaptions
from transformers import BlipProcessor, BlipModel
from tqdm import tqdm


# ======================================
# Config
# ======================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Paragraph JSONs (from Qwen+Llama+SPICE)
VAL_PARAGRAPH_JSON = "/home/mahyarghazanfari/workspace/Project/Dataset/generated_coco_validation_qwen_llama_with_spice.json"

# COCO val (for multi-caption evaluation)
VAL_ROOT = "data/coco/validation/val2017"
VAL_ANN  = "data/coco/annotations/captions_val2017.json"

# Checkpoints
# 1) BLIP fine-tuned on multi-positive captions
MULTIPOS_CKPT_DIR   = "/home/mahyarghazanfari/workspace/Project/checkpoints_multipositive/blip_multipos_finetuned_epoch_20"

# 2) BLIP fine-tuned on paragraphs
PARAGRAPH_CKPT_DIR  = "checkpoints_paragraphs/blip_paragraph_finetuned_epoch_10"

CAPS_PER_IMG = 5  # number of COCO captions per image to use


# ======================================
# Paragraph dataset
# ======================================
class CocoParagraphDataset(Dataset):
    """
    Dataset that reads from the Qwen+Llama paragraph JSON and returns (image, paragraph).
    Expected JSON fields:
      - "image_path"
      - "paragraph"
    Optional:
      - "llama_score", "spice", etc. (ignored here)
    """
    def __init__(self, json_path: str):
        super().__init__()
        assert os.path.exists(json_path), f"JSON file not found: {json_path}"

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries: List[Dict[str, Any]] = []
        for item in data:
            paragraph = item.get("paragraph", None)
            if paragraph is None or not str(paragraph).strip():
                continue

            img_path = item.get("image_path", None)
            if img_path is None or not os.path.exists(img_path):
                continue

            entries.append(
                {
                    "image_path": img_path,
                    "paragraph": paragraph.strip(),
                    "image_id": item.get("image_id", None),
                }
            )

        self.entries = entries
        print(f"[Paragraph val] Loaded {len(self.entries)} image-paragraph pairs from {json_path}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        paragraph = item["paragraph"]
        return image, paragraph


# ======================================
# COCO multi-caption dataset (val)
# ======================================
class CocoAllCaptionsDataset(Dataset):
    """
    For val COCO, returns (image, captions_list) with up to caps_per_img captions.
    """
    def __init__(self, coco: CocoCaptions, caps_per_img: int = 5):
        self.coco = coco
        self.indices = list(range(len(coco)))
        self.caps_per_img = caps_per_img

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_idx = self.indices[idx]
        image, captions = self.coco[img_idx]
        captions = list(captions)[: self.caps_per_img]
        if len(captions) == 0:
            raise ValueError(f"Image at index {img_idx} has no captions.")
        return image, captions


# ======================================
# Recall helpers
# ======================================
def recall_at_k_one_positive(sim_matrix: torch.Tensor, k: int) -> float:
    """
    One positive per row/column.
    Row i's positive index = i.
    """
    M = sim_matrix.size(0)
    correct = 0
    for i in range(M):
        sims = sim_matrix[i]
        topk_idx = sims.topk(k).indices.tolist()
        if i in topk_idx:
            correct += 1
    return correct / M


def recall_at_k_i2t_multi(sim_matrix: torch.Tensor, k: int, caps_per_img: int = 5) -> float:
    """
    Image->Text recall for multi-caption setup.
    Image i's positives are text indices [caps_per_img*i ... caps_per_img*i + caps_per_img - 1].
    """
    M = sim_matrix.size(0)
    correct = 0
    for i in range(M):
        sims = sim_matrix[i]
        topk_idx = sims.topk(k).indices.tolist()
        true_indices = set(range(caps_per_img * i, caps_per_img * i + caps_per_img))
        if any(idx in true_indices for idx in topk_idx):
            correct += 1
    return correct / M


def recall_at_k_t2i_multi(sim_matrix: torch.Tensor, k: int, caps_per_img: int = 5) -> float:
    """
    Text->Image recall for multi-caption setup.
    Caption index t's positive image = t // caps_per_img.
    """
    N_txt = sim_matrix.size(0)
    correct = 0
    for caption_idx in range(N_txt):
        sims = sim_matrix[caption_idx]
        topk_idx = sims.topk(k).indices.tolist()
        true_img_idx = caption_idx // caps_per_img
        if true_img_idx in topk_idx:
            correct += 1
    return correct / N_txt


# ======================================
# Embedding functions
# ======================================
@torch.no_grad()
def compute_embeddings_paragraph_eval(model, processor, dataset: Dataset):
    """
    Paragraph-only setup.
    For each item: (image, paragraph).
    Returns:
      image_features: [M, D]
      text_features:  [M, D]
    """
    model.eval()
    image_features = []
    text_features = []

    for idx in tqdm(range(len(dataset)), desc="Encoding val paragraphs"):
        image, paragraph = dataset[idx]

        img_inputs = processor(images=image, return_tensors="pt").to(device)
        txt_inputs = processor(text=[paragraph], return_tensors="pt", padding=True).to(device)

        img_emb = model.get_image_features(**img_inputs)[0]  # [D]
        txt_emb = model.get_text_features(
            input_ids=txt_inputs.input_ids,
            attention_mask=txt_inputs.attention_mask,
        )[0]  # [D]

        image_features.append(img_emb.cpu())
        text_features.append(txt_emb.cpu())

    image_features = torch.stack(image_features, dim=0)
    text_features = torch.stack(text_features, dim=0)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return image_features, text_features


@torch.no_grad()
def compute_embeddings_captions_eval(model, processor, coco_dataset: CocoCaptions, caps_per_img: int = 5):
    """
    Multi-caption setup for COCO.
    Returns:
      image_features: [M, D]
      text_features:  [M * caps_per_img, D]
    """
    model.eval()
    ds = CocoAllCaptionsDataset(coco_dataset, caps_per_img=caps_per_img)

    image_features = []
    text_features = []

    for idx in tqdm(range(len(ds)), desc="Encoding val captions"):
        image, captions = ds[idx]
        captions = list(captions)[:caps_per_img]
        if len(captions) < caps_per_img:
            continue

        img_inputs = processor(images=image, return_tensors="pt").to(device)
        img_emb = model.get_image_features(**img_inputs)[0]
        image_features.append(img_emb.cpu())

        txt_inputs = processor(text=captions, return_tensors="pt", padding=True).to(device)
        txt_embs = model.get_text_features(
            input_ids=txt_inputs.input_ids,
            attention_mask=txt_inputs.attention_mask,
        )
        text_features.append(txt_embs.cpu())

    image_features = torch.stack(image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return image_features, text_features


# ======================================
# 1) Multi-caption model → Paragraphs
# ======================================
print("\n=== 1) Evaluating MULTI-CAPTION model on PARAGRAPHS (val) ===")

# Load paragraph val dataset
val_paragraph_dataset = CocoParagraphDataset(VAL_PARAGRAPH_JSON)

# Load multi-caption fine-tuned BLIP
multi_model = BlipModel.from_pretrained(MULTIPOS_CKPT_DIR).to(device)
multi_processor = BlipProcessor.from_pretrained(MULTIPOS_CKPT_DIR)

# Compute embeddings
img_feats_para, txt_feats_para = compute_embeddings_paragraph_eval(
    multi_model, multi_processor, val_paragraph_dataset
)

print("Paragraph eval - image_feats:", img_feats_para.shape)
print("Paragraph eval - text_feats:", txt_feats_para.shape)

# Similarity and recall
sim_para = img_feats_para @ txt_feats_para.T
sim_para_t2i = sim_para.T

print("\n[Multi-caption model] Image-to-Paragraph retrieval (one positive):")
for k in [1, 5, 10]:
    r = recall_at_k_one_positive(sim_para, k)
    print(f"R@{k}: {r:.4f}")

print("\n[Multi-caption model] Paragraph-to-Image retrieval (one positive):")
for k in [1, 5, 10]:
    r = recall_at_k_one_positive(sim_para_t2i, k)
    print(f"R@{k}: {r:.4f}")


# ======================================
# 2) Paragraph model → Multi-captions
# ======================================
print("\n=== 2) Evaluating PARAGRAPH model on MULTI-CAPTIONS (COCO val) ===")

# Load COCO val
val_coco = CocoCaptions(root=VAL_ROOT, annFile=VAL_ANN, transform=None)
print("Number of val images:", len(val_coco))

# Load paragraph fine-tuned BLIP
para_model = BlipModel.from_pretrained(PARAGRAPH_CKPT_DIR).to(device)
para_processor = BlipProcessor.from_pretrained(PARAGRAPH_CKPT_DIR)

# Compute embeddings with 5 captions per image
img_feats_cap, txt_feats_cap = compute_embeddings_captions_eval(
    para_model, para_processor, val_coco, caps_per_img=CAPS_PER_IMG
)

M = img_feats_cap.size(0)
N_txt = txt_feats_cap.size(0)
print("Caption eval - image_feats:", img_feats_cap.shape)
print("Caption eval - text_feats:", txt_feats_cap.shape)
assert N_txt == M * CAPS_PER_IMG, f"Expected {M * CAPS_PER_IMG}, got {N_txt}"

sim_cap = img_feats_cap @ txt_feats_cap.T      # [M, 5M]
sim_cap_t2i = sim_cap.T                        # [5M, M]

print("\n[Paragraph model] Image-to-Text (captions) retrieval (multi-positive):")
for k in [1, 5, 10]:
    r = recall_at_k_i2t_multi(sim_cap, k, caps_per_img=CAPS_PER_IMG)
    print(f"R@{k}: {r:.4f}")

print("\n[Paragraph model] Text-to-Image (captions) retrieval (multi-positive):")
for k in [1, 5, 10]:
    r = recall_at_k_t2i_multi(sim_cap_t2i, k, caps_per_img=CAPS_PER_IMG)
    print(f"R@{k}: {r:.4f}")
