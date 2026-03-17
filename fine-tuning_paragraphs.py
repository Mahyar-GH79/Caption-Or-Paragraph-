import os
import json
from typing import List, Dict, Any

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import BlipProcessor, BlipModel
from tqdm import tqdm
import matplotlib.pyplot as plt


# ======================================
# Config
# ======================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Paths to your generated JSONs (from Qwen + Llama + SPICE)
TRAIN_JSON = "/home/mahyarghazanfari/workspace/Project/Dataset/generated_coco_train_qwen_llama_with_spice.json"
VAL_JSON   = "/home/mahyarghazanfari/workspace/Project/Dataset/generated_coco_validation_qwen_llama_with_spice.json"

batch_size = 32
num_epochs = 10
learning_rate = 1e-5
weight_decay = 1e-4
temperature = 0.07

min_llama_score = None   # e.g. 5 if you want to keep only high-scoring paragraphs
min_spice = None         # e.g. 0.3 if you want to keep only semantically faithful ones

os.makedirs("checkpoints_paragraphs", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)


# ======================================
# Dataset
# ======================================
class CocoParagraphDataset(Dataset):
    """
    Dataset that reads from the Qwen+Llama paragraph JSON and returns (image, paragraph).
    Each entry in JSON is expected to have:
      - "image_path": path to the image
      - "paragraph": generated paragraph (may be None)
      - "llama_score": optional numeric
      - "spice": optional numeric
    """
    def __init__(
        self,
        json_path: str,
        split_name: str,
        min_llama_score: float = None,
        min_spice: float = None,
    ):
        super().__init__()
        assert os.path.exists(json_path), f"JSON file not found: {json_path}"
        self.split_name = split_name

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries: List[Dict[str, Any]] = []
        for item in data:
            # Basic checks
            if "paragraph" not in item:
                continue
            paragraph = item["paragraph"]
            if paragraph is None or not str(paragraph).strip():
                continue

            # Optional filters based on Llama score
            if min_llama_score is not None:
                s = item.get("llama_score", None)
                if s is None or s < min_llama_score:
                    continue

            # Optional filters based on SPICE
            if min_spice is not None:
                s = item.get("spice", None)
                if s is None or s < min_spice:
                    continue

            img_path = item.get("image_path", None)
            if img_path is None or not os.path.exists(img_path):
                # You can print a warning or skip
                # print(f"Warning: image path missing or not found: {img_path}")
                continue

            entries.append(
                {
                    "image_path": img_path,
                    "paragraph": paragraph.strip(),
                    "image_id": item.get("image_id", None),
                }
            )

        self.entries = entries
        print(
            f"[{split_name}] Loaded {len(self.entries)} usable image-paragraph pairs "
            f"from {json_path}"
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        image_path = item["image_path"]
        paragraph = item["paragraph"]

        image = Image.open(image_path).convert("RGB")
        return image, paragraph


def collate_fn_pil(batch):
    """
    Custom collate so DataLoader does not try to stack PIL images.
    batch is list of (image, paragraph).
    """
    images, paragraphs = zip(*batch)
    return list(images), list(paragraphs)


# ======================================
# Load Datasets and DataLoaders
# ======================================
train_dataset = CocoParagraphDataset(
    TRAIN_JSON,
    split_name="train",
    min_llama_score=min_llama_score,
    min_spice=min_spice,
)

val_dataset = CocoParagraphDataset(
    VAL_JSON,
    split_name="val",
    min_llama_score=min_llama_score,
    min_spice=min_spice,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn_pil,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn_pil,
)


# ======================================
# Load BLIP model and processor
# ======================================
model_name = "Salesforce/blip-itm-base-coco"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipModel.from_pretrained(model_name).to(device)

# Freeze image encoder (vision_model and projection)
for param in model.vision_model.parameters():
    param.requires_grad = False
if hasattr(model, "visual_projection"):
    for param in model.visual_projection.parameters():
        param.requires_grad = False

# Keep text_model and text_projection trainable
trainable_params = [p for p in model.parameters() if p.requires_grad]
print(
    "Number of trainable parameters (paragraph fine tuning):",
    sum(p.numel() for p in trainable_params),
)

optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)


# ======================================
# Contrastive loss (InfoNCE, one positive per image here)
# ======================================
def contrastive_loss(image_feats, text_feats, temperature=0.07):
    """
    Symmetric InfoNCE loss for one-positive-per-row/column training.
    image_feats: [B, D]
    text_feats: [B, D]
    """
    image_feats = nn.functional.normalize(image_feats, dim=-1)
    text_feats = nn.functional.normalize(text_feats, dim=-1)

    logits = image_feats @ text_feats.T  # [B, B]
    logits = logits / temperature

    batch_size = logits.size(0)
    labels = torch.arange(batch_size, device=logits.device)

    loss_i2t = nn.functional.cross_entropy(logits, labels)
    loss_t2i = nn.functional.cross_entropy(logits.T, labels)
    loss = (loss_i2t + loss_t2i) / 2
    return loss


# ======================================
# Training loop (train on train paragraphs, validate on val paragraphs)
# ======================================
val_losses = []

for epoch in range(num_epochs):
    # Train
    model.train()
    total_loss = 0.0
    num_batches = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]")

    for images, paragraphs in progress:
        # Build inputs for BLIP
        inputs = processor(
            images=images,
            text=paragraphs,
            return_tensors="pt",
            padding=True,
        ).to(device)

        image_feats = model.get_image_features(pixel_values=inputs.pixel_values)
        text_feats = model.get_text_features(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )

        loss = contrastive_loss(image_feats, text_feats, temperature=temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        progress.set_postfix({"train_loss": total_loss / num_batches})

    avg_train_loss = total_loss / num_batches
    print(f"Epoch {epoch+1} finished, avg train loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_total = 0.0
    val_batches = 0
    with torch.no_grad():
        for images, paragraphs in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val]"):
            inputs = processor(
                images=images,
                text=paragraphs,
                return_tensors="pt",
                padding=True,
            ).to(device)

            image_feats = model.get_image_features(pixel_values=inputs.pixel_values)
            text_feats = model.get_text_features(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )

            val_loss = contrastive_loss(image_feats, text_feats, temperature=temperature)
            val_total += val_loss.item()
            val_batches += 1

    avg_val_loss = val_total / val_batches
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch+1}, avg val loss (paragraphs): {avg_val_loss:.4f}")

    # Save checkpoint after each epoch
    ckpt_dir = f"checkpoints_paragraphs/blip_paragraph_finetuned_epoch_{epoch+1}"
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Saving checkpoint to {ckpt_dir}")
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)


# ======================================
# Plot validation loss
# ======================================
plt.figure()
plt.plot(range(1, num_epochs + 1), val_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Validation loss (paragraphs)")
plt.title("Validation loss over epochs (COCO paragraphs)")
plt.grid(True)
plt.savefig("plots/val_loss_paragraphs.png", dpi=200)
plt.close()
print("Saved validation loss plot to plots/val_loss_paragraphs.png")


# ======================================
# Retrieval evaluation on val paragraphs
# ======================================
@torch.no_grad()
def compute_embeddings_for_eval(model, processor, dataset):
    """
    For the paragraph-only setting, each image has exactly one paragraph (after filtering).
    Returns:
      image_features: [M, D]
      text_features:  [M, D]
    """
    image_features = []
    text_features = []

    for idx in tqdm(range(len(dataset)), desc="Encoding val paragraphs for retrieval"):
        image, paragraph = dataset[idx]

        img_inputs = processor(images=image, return_tensors="pt").to(device)
        txt_inputs = processor(text=[paragraph], return_tensors="pt", padding=True).to(device)

        img_emb = model.get_image_features(**img_inputs)[0]          # [D]
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


image_feats_val, text_feats_val = compute_embeddings_for_eval(
    model, processor, val_dataset
)

M = image_feats_val.size(0)
print("Val image_features:", image_feats_val.shape)
print("Val text_features:", text_feats_val.shape)

# similarity matrices
similarity = image_feats_val @ text_feats_val.T
similarity_t2i = similarity.T


def recall_at_k_one_positive(sim_matrix, k):
    """
    For the paragraph-only setup, there is exactly one positive per row/column.
    Row i corresponds to image i, column j to paragraph j.
    Positive is j == i.
    """
    M = sim_matrix.size(0)
    correct = 0
    for i in range(M):
        sims = sim_matrix[i]
        topk_idx = sims.topk(k).indices.tolist()
        if i in topk_idx:
            correct += 1
    return correct / M


print("\nVal Image-to-Paragraph (I2T with paragraphs) retrieval:")
for k in [1, 5, 10]:
    r = recall_at_k_one_positive(similarity, k)
    print(f"R@{k}: {r:.4f}")

print("\nVal Paragraph-to-Image (T2I with paragraphs) retrieval:")
for k in [1, 5, 10]:
    r = recall_at_k_one_positive(similarity_t2i, k)
    print(f"R@{k}: {r:.4f}")

# Save embeddings for later
emb_path = "embeddings/blip_val_paragraphs_embeddings.pt"
torch.save(
    {
        "image_feats_val": image_feats_val,
        "text_feats_val": text_feats_val,
        "setting": "paragraph_only",
    },
    emb_path,
)
print(f"Saved val paragraph embeddings to {emb_path}")
