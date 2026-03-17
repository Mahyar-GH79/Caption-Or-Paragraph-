import os
import json
from typing import List, Dict, Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoCaptions
from PIL import Image
from transformers import BlipProcessor, BlipModel
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# COCO paths
train_root = "data/coco/train/train2017"
train_ann  = "data/coco/annotations/captions_train2017.json"

val_root = "data/coco/validation/val2017"
val_ann  = "data/coco/annotations/captions_val2017.json"

# Paragraph JSONs (from Qwen + Llama + SPICE)
TRAIN_JSON = "/home/mahyarghazanfari/workspace/Project/Dataset/generated_coco_train_qwen_llama_with_spice.json"
VAL_JSON   = "/home/mahyarghazanfari/workspace/Project/Dataset/generated_coco_validation_qwen_llama_with_spice.json"

# Filters for paragraphs (optional; set to None to disable)
min_llama_score = None   # e.g. 5
min_spice       = None   # e.g. 0.3

caps_per_img   = 5       # number of COCO captions per image
batch_size     = 24      # number of images per batch
num_epochs     = 10
learning_rate  = 1e-5
weight_decay   = 1e-4
temperature    = 0.07

os.makedirs("checkpoints_combined", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

# ============================================================
# Utilities: load paragraph mappings
# ============================================================

def load_paragraph_map(
    json_path: str,
    min_llama_score: float = None,
    min_spice: float = None,
) -> Dict[int, str]:
    """
    Build a mapping from image_id -> paragraph string based on the generated JSON.
    Filters by optional Llama score and SPICE.
    """
    assert os.path.exists(json_path), f"Paragraph JSON not found: {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_paragraph: Dict[int, str] = {}
    for item in data:
        para = item.get("paragraph", None)
        if para is None or not str(para).strip():
            continue

        if min_llama_score is not None:
            s = item.get("llama_score", None)
            if s is None or s < min_llama_score:
                continue

        if min_spice is not None:
            s = item.get("spice", None)
            if s is None or s < min_spice:
                continue

        img_id = item.get("image_id", None)
        if img_id is None:
            continue

        id_to_paragraph[int(img_id)] = para.strip()

    print(
        f"Loaded {len(id_to_paragraph)} image_id -> paragraph entries "
        f"from {json_path} (filters: llama>={min_llama_score}, spice>={min_spice})"
    )
    return id_to_paragraph


# ============================================================
# Datasets
# ============================================================

class CocoAllCaptionsDataset(Dataset):
    """
    For captions-only evaluation:
    Each __getitem__ returns (image, [up to caps_per_img captions]).
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
        return image, captions


class CocoCombinedCaptionsParagraphDataset(Dataset):
    """
    Combined dataset for training or validation:

      - Uses COCO captions (up to caps_per_img per image).
      - Adds 1 paragraph from id_to_paragraph for that image_id (if available).

    Each __getitem__ returns:
        image, combined_texts

      where combined_texts is [cap1, cap2, ..., capK, paragraph]
    """
    def __init__(
        self,
        coco: CocoCaptions,
        id_to_paragraph: Dict[int, str],
        split_name: str,
        caps_per_img: int = 5,
    ):
        super().__init__()
        self.coco = coco
        self.ids = coco.ids  # list of COCO image ids
        self.id_to_paragraph = id_to_paragraph
        self.caps_per_img = caps_per_img

        # We only keep indices for which we have a paragraph
        indices = []
        for idx, img_id in enumerate(self.ids):
            if int(img_id) in self.id_to_paragraph:
                indices.append(idx)

        self.indices = indices
        print(
            f"[{split_name} combined] Using {len(self.indices)} images "
            f"that have both COCO captions and a paragraph."
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]  # index in original COCO dataset
        img_id = int(self.ids[real_idx])

        image, captions = self.coco[real_idx]
        captions = list(captions)[: self.caps_per_img]
        if len(captions) == 0:
            raise ValueError(f"Image id {img_id} has no COCO captions.")

        paragraph = self.id_to_paragraph[img_id]

        texts = captions + [paragraph]
        return image, texts


class ParagraphValDataset(Dataset):
    """
    Paragraph-only dataset for validation retrieval:

      Each entry from VAL_JSON should have image_path + paragraph.
      We do not require alignment with COCO dataset here; each pair is independent.

    Returns (image, paragraph).
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

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries: List[Dict[str, Any]] = []
        for item in data:
            paragraph = item.get("paragraph", None)
            if paragraph is None or not str(paragraph).strip():
                continue

            if min_llama_score is not None:
                s = item.get("llama_score", None)
                if s is None or s < min_llama_score:
                    continue

            if min_spice is not None:
                s = item.get("spice", None)
                if s is None or s < min_spice:
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
        print(
            f"[{split_name} paragraphs] Loaded {len(self.entries)} usable "
            f"image-paragraph pairs from {json_path}"
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        img_path = item["image_path"]
        paragraph = item["paragraph"]

        image = Image.open(img_path).convert("RGB")
        return image, paragraph


# Custom collates
def collate_fn_pil_all(batch):
    """
    batch: list of (image, texts_list)
    returns:
        images: list of PIL images
        texts_per_image: list of list[str]
    """
    images, texts_per_image = zip(*batch)
    return list(images), [list(t) for t in texts_per_image]


def collate_fn_paragraph(batch):
    images, paragraphs = zip(*batch)
    return list(images), list(paragraphs)


# ============================================================
# Load COCO and paragraph maps
# ============================================================
train_coco = CocoCaptions(root=train_root, annFile=train_ann, transform=None)
val_coco   = CocoCaptions(root=val_root,  annFile=val_ann,  transform=None)

print("Number of train images:", len(train_coco))
print("Number of val images:", len(val_coco))

train_para_map = load_paragraph_map(
    TRAIN_JSON,
    min_llama_score=min_llama_score,
    min_spice=min_spice,
)
val_para_map = load_paragraph_map(
    VAL_JSON,
    min_llama_score=min_llama_score,
    min_spice=min_spice,
)

# Combined train/val datasets for training and val loss
train_dataset_combined = CocoCombinedCaptionsParagraphDataset(
    train_coco,
    id_to_paragraph=train_para_map,
    split_name="train",
    caps_per_img=caps_per_img,
)

val_dataset_combined = CocoCombinedCaptionsParagraphDataset(
    val_coco,
    id_to_paragraph=val_para_map,
    split_name="val",
    caps_per_img=caps_per_img,
)

train_loader = DataLoader(
    train_dataset_combined,
    batch_size=batch_size,      # images
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn_pil_all,
)

val_loader = DataLoader(
    val_dataset_combined,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn_pil_all,
)

# Caption-only val dataset for retrieval
val_captions_dataset = CocoAllCaptionsDataset(
    val_coco,
    caps_per_img=caps_per_img,
)

# Paragraph-only val dataset for retrieval
val_paragraph_dataset = ParagraphValDataset(
    VAL_JSON,
    split_name="val_paragraphs",
    min_llama_score=min_llama_score,
    min_spice=min_spice,
)


# ============================================================
# Load BLIP model and processor
# ============================================================
model_name = "Salesforce/blip-itm-base-coco"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipModel.from_pretrained(model_name).to(device)

# Freeze image encoder, fine tune text side
for param in model.vision_model.parameters():
    param.requires_grad = False
if hasattr(model, "visual_projection"):
    for param in model.visual_projection.parameters():
        param.requires_grad = False

trainable_params = [p for p in model.parameters() if p.requires_grad]
print(
    "Number of trainable parameters (combined captions+paragraphs):",
    sum(p.numel() for p in trainable_params),
)

optimizer = torch.optim.AdamW(
    trainable_params,
    lr=learning_rate,
    weight_decay=weight_decay,
)


# ============================================================
# Multi positive contrastive loss (image with many captions+paragraph)
# ============================================================
def multi_positive_contrastive_loss(
    image_feats: torch.Tensor,              # [B, D]
    text_feats: torch.Tensor,               # [T, D]
    img_indices_for_text: torch.Tensor,     # [T] each in [0, B-1]
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Multi positive symmetric InfoNCE loss.

    image_feats: [B, D] embeddings for B images
    text_feats:  [T, D] embeddings for all captions/paragraphs in the batch
    img_indices_for_text: [T] mapping each text to its image index
    """
    I = F.normalize(image_feats, dim=-1)
    T = F.normalize(text_feats, dim=-1)

    B = I.size(0)
    Tn = T.size(0)
    tau = temperature

    # image -> text
    logits_i2t = (I @ T.T) / tau           # [B, Tn]
    img_indices_for_text = img_indices_for_text.to(I.device)
    img_row = torch.arange(B, device=I.device).unsqueeze(1)        # [B, 1]
    txt_col = img_indices_for_text.unsqueeze(0)                    # [1, Tn]
    pos_mask = (img_row == txt_col)                                # [B, Tn] bool

    logsumexp_all = torch.logsumexp(logits_i2t, dim=1)             # [B]
    logits_pos = logits_i2t.masked_fill(~pos_mask, float("-inf"))
    logsumexp_pos = torch.logsumexp(logits_pos, dim=1)             # [B]

    valid = torch.isfinite(logsumexp_pos)
    loss_i2t = -(logsumexp_pos[valid] - logsumexp_all[valid]).mean()

    # text -> image (one positive per text)
    logits_t2i = (T @ I.T) / tau                                   # [Tn, B]
    labels_t = img_indices_for_text                                # [Tn]
    loss_t2i = F.cross_entropy(logits_t2i, labels_t)

    return (loss_i2t + loss_t2i) / 2.0


# ============================================================
# Training loop with validation loss (combined)
# ============================================================
val_losses = []

for epoch in range(num_epochs):
    # 1) Train
    model.train()
    total_loss = 0.0
    num_batches = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]")

    for images, texts_per_image in progress:
        # Flatten all texts and build mapping
        flat_texts: List[str] = []
        img_indices_for_text: List[int] = []

        for img_idx_in_batch, texts in enumerate(texts_per_image):
            for t in texts:
                flat_texts.append(t)
                img_indices_for_text.append(img_idx_in_batch)

        img_indices_for_text = torch.tensor(
            img_indices_for_text,
            dtype=torch.long,
            device=device,
        )

        inputs = processor(
            images=images,
            text=flat_texts,
            return_tensors="pt",
            padding=True,
        ).to(device)

        image_feats = model.get_image_features(pixel_values=inputs.pixel_values)  # [B, D]
        text_feats = model.get_text_features(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )  # [T, D]

        loss = multi_positive_contrastive_loss(
            image_feats,
            text_feats,
            img_indices_for_text,
            temperature=temperature,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        progress.set_postfix({"train_loss": total_loss / num_batches})

    avg_train_loss = total_loss / num_batches
    print(f"Epoch {epoch+1} finished, avg train loss: {avg_train_loss:.4f}")

    # 2) Validation loss on combined val (captions + paragraph)
    model.eval()
    val_total = 0.0
    val_batches = 0

    with torch.no_grad():
        for images, texts_per_image in tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{num_epochs} [val_combined]",
        ):
            flat_texts: List[str] = []
            img_indices_for_text: List[int] = []

            for img_idx_in_batch, texts in enumerate(texts_per_image):
                for t in texts:
                    flat_texts.append(t)
                    img_indices_for_text.append(img_idx_in_batch)

            img_indices_for_text = torch.tensor(
                img_indices_for_text,
                dtype=torch.long,
                device=device,
            )

            inputs = processor(
                images=images,
                text=flat_texts,
                return_tensors="pt",
                padding=True,
            ).to(device)

            image_feats = model.get_image_features(pixel_values=inputs.pixel_values)
            text_feats = model.get_text_features(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )

            val_loss = multi_positive_contrastive_loss(
                image_feats,
                text_feats,
                img_indices_for_text,
                temperature=temperature,
            )
            val_total += val_loss.item()
            val_batches += 1

    avg_val_loss = val_total / val_batches
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch+1}, avg combined val loss: {avg_val_loss:.4f}")

    # Save checkpoint
    ckpt_dir = f"checkpoints_combined/blip_combined_caps_para_epoch_{epoch+1}"
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Saving checkpoint to {ckpt_dir}")
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)

# Switch to eval for final retrieval
model.eval()

# ============================================================
# Plot validation loss
# ============================================================
plt.figure()
plt.plot(range(1, num_epochs + 1), val_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Combined val loss (captions + paragraph)")
plt.title("Validation loss over epochs (combined training)")
plt.grid(True)
plt.savefig("plots/val_loss_combined_caps_plus_para.png", dpi=200)
plt.close()
print("Saved validation loss plot to plots/val_loss_combined_caps_plus_para.png")


# ============================================================
# Retrieval eval: COCO val captions (multi positive)
# ============================================================
@torch.no_grad()
def compute_embeddings_for_eval_captions(model, processor, dataset, caps_per_img=5):
    """
    For captions:
      Returns image_features: [M, D]
              text_features:  [M * caps_per_img, D]
    """
    image_features = []
    text_features = []

    for idx in tqdm(range(len(dataset)), desc="Encoding val captions for retrieval"):
        image, captions = dataset[idx]
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

    image_features = torch.stack(image_features, dim=0)   # [M, D]
    text_features = torch.cat(text_features, dim=0)       # [M * caps_per_img, D]

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return image_features, text_features


def recall_at_k_i2t_multi(sim_matrix, k, caps_per_img=5):
    """
    Image to text recall with multiple positives per image.
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


def recall_at_k_t2i_multi(sim_matrix, k, caps_per_img=5):
    """
    Text to image recall with multiple positives per image.
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


image_feats_val_caps, text_feats_val_caps = compute_embeddings_for_eval_captions(
    model,
    processor,
    val_captions_dataset,
    caps_per_img=caps_per_img,
)

print("Val captions image_features:", image_feats_val_caps.shape)
print("Val captions text_features:", text_feats_val_caps.shape)

similarity_caps = image_feats_val_caps @ text_feats_val_caps.T
similarity_caps_t2i = similarity_caps.T

print("\nVal Image-to-Text (I2T) retrieval on COCO val captions:")
for k in [1, 5, 10]:
    r = recall_at_k_i2t_multi(similarity_caps, k, caps_per_img=caps_per_img)
    print(f"R@{k}: {r:.4f}")

print("\nVal Text-to-Image (T2I) retrieval on COCO val captions:")
for k in [1, 5, 10]:
    r = recall_at_k_t2i_multi(similarity_caps_t2i, k, caps_per_img=caps_per_img)
    print(f"R@{k}: {r:.4f}")

torch.save(
    {
        "image_feats_val_caps": image_feats_val_caps,
        "text_feats_val_caps": text_feats_val_caps,
        "caps_per_img": caps_per_img,
    },
    "embeddings/blip_combined_val_captions_embeddings.pt",
)
print("Saved val captions embeddings to embeddings/blip_combined_val_captions_embeddings.pt")


# ============================================================
# Retrieval eval: val paragraphs (one positive per image)
# ============================================================
@torch.no_grad()
def compute_embeddings_for_eval_paragraphs(model, processor, dataset):
    """
    For paragraph retrieval:
      each item is (image, paragraph).
      Returns image_features: [M, D]
              text_features:  [M, D]
    """
    image_features = []
    text_features = []

    for idx in tqdm(range(len(dataset)), desc="Encoding val paragraphs for retrieval"):
        image, paragraph = dataset[idx]

        img_inputs = processor(images=image, return_tensors="pt").to(device)
        txt_inputs = processor(text=[paragraph], return_tensors="pt", padding=True).to(device)

        img_emb = model.get_image_features(**img_inputs)[0]
        txt_emb = model.get_text_features(
            input_ids=txt_inputs.input_ids,
            attention_mask=txt_inputs.attention_mask,
        )[0]

        image_features.append(img_emb.cpu())
        text_features.append(txt_emb.cpu())

    image_features = torch.stack(image_features, dim=0)
    text_features = torch.stack(text_features, dim=0)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return image_features, text_features


def recall_at_k_one_positive(sim_matrix, k):
    """
    For paragraphs: exactly one positive per row/column.
    Row i corresponds to image i, column i to its paragraph.
    """
    M = sim_matrix.size(0)
    correct = 0
    for i in range(M):
        sims = sim_matrix[i]
        topk_idx = sims.topk(k).indices.tolist()
        if i in topk_idx:
            correct += 1
    return correct / M


image_feats_val_para, text_feats_val_para = compute_embeddings_for_eval_paragraphs(
    model,
    processor,
    val_paragraph_dataset,
)

print("Val paragraphs image_features:", image_feats_val_para.shape)
print("Val paragraphs text_features:", text_feats_val_para.shape)

similarity_para = image_feats_val_para @ text_feats_val_para.T
similarity_para_t2i = similarity_para.T

print("\nVal Image-to-Paragraph retrieval:")
for k in [1, 5, 10]:
    r = recall_at_k_one_positive(similarity_para, k)
    print(f"R@{k}: {r:.4f}")

print("\nVal Paragraph-to-Image retrieval:")
for k in [1, 5, 10]:
    r = recall_at_k_one_positive(similarity_para_t2i, k)
    print(f"R@{k}: {r:.4f}")

torch.save(
    {
        "image_feats_val_para": image_feats_val_para,
        "text_feats_val_para": text_feats_val_para,
        "setting": "combined_train_eval_on_paragraphs",
    },
    "embeddings/blip_combined_val_paragraphs_embeddings.pt",
)
print("Saved val paragraph embeddings to embeddings/blip_combined_val_paragraphs_embeddings.pt")
