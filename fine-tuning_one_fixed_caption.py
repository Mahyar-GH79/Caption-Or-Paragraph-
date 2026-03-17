import os
from typing import List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoCaptions
from transformers import BlipProcessor, BlipModel
from tqdm import tqdm
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Paths
train_root = "data/coco/train/train2017"
train_ann = "data/coco/annotations/captions_train2017.json"

val_root = "data/coco/validation/val2017"
val_ann = "data/coco/annotations/captions_val2017.json"

# 1. Load COCO train and val
train_coco = CocoCaptions(root=train_root, annFile=train_ann, transform=None)
val_coco = CocoCaptions(root=val_root, annFile=val_ann, transform=None)

num_train_images = len(train_coco)
num_val_images = len(val_coco)

print("Number of train images:", num_train_images)
print("Number of val images:", num_val_images)


# 2. Dataset wrapper for training: returns (image, one FIXED caption per image)
class CocoFixedCaptionDataset(Dataset):
    """
    For each image, deterministically pick one caption (by index)
    and always return that same caption during training or validation.
    """
    def __init__(
        self,
        coco: CocoCaptions,
        caps_per_img: int = 5,
        caption_index: int = 0,  # which caption to pick (0 = first)
    ):
        self.coco = coco
        self.caps_per_img = caps_per_img
        self.indices = []          # indices into coco
        self.fixed_captions = []   # fixed caption per entry

        for img_idx in range(len(coco)):
            image, captions = self.coco[img_idx]
            captions = list(captions)[: self.caps_per_img]
            if len(captions) == 0:
                # skip images with no captions (should not happen for COCO)
                continue

            # Clamp caption_index in case some images have fewer captions
            idx = min(caption_index, len(captions) - 1)
            chosen_caption = captions[idx]

            self.indices.append(img_idx)
            self.fixed_captions.append(chosen_caption)

        print(
            f"CocoFixedCaptionDataset built with {len(self.indices)} images "
            f"and one fixed caption per image (caption index {caption_index})."
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_idx = self.indices[idx]
        image, _ = self.coco[img_idx]  # ignore original captions here
        caption = self.fixed_captions[idx]
        return image, caption


# 3. Dataset wrapper for evaluation: use all captions (capped at 5) per image
class CocoAllCaptionsDataset(Dataset):
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


# Custom collate function so DataLoader does not try to stack PIL images
def collate_fn_pil(batch):
    # batch is list of (image, caption)
    images, captions = zip(*batch)
    return list(images), list(captions)


caps_per_img = 5

# 4. Create datasets and loaders
# Use fixed caption index 0 (first caption) for both train and val
train_dataset = CocoFixedCaptionDataset(
    train_coco,
    caps_per_img=caps_per_img,
    caption_index=0,
)
val_dataset = CocoFixedCaptionDataset(
    val_coco,
    caps_per_img=caps_per_img,
    caption_index=0,
)
test_dataset = CocoAllCaptionsDataset(val_coco, caps_per_img=caps_per_img)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn_pil,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn_pil,
)
# test_dataset will be used in a simple loop, not batched


# 5. Load BLIP model and processor
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
print("Number of trainable parameters:", sum(p.numel() for p in trainable_params))

optimizer = torch.optim.AdamW(trainable_params, lr=1e-5, weight_decay=1e-4)

temperature = 0.07


def contrastive_loss(image_feats, text_feats, temperature=0.07):
    """
    Symmetric InfoNCE loss.
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


# 6. Training loop with validation loss tracking (train on train2017, val on val2017)
num_epochs = 10
val_losses = []

os.makedirs("checkpoints_onecap", exist_ok=True)

for epoch in range(num_epochs):
    # Train
    model.train()
    total_loss = 0.0
    num_batches = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]")
    for images, captions in progress:
        inputs = processor(
            images=images,
            text=captions,
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

    # Validation on full val2017 (one fixed caption per image)
    model.eval()
    val_total = 0.0
    val_batches = 0

    with torch.no_grad():
        for images, captions in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val]"):
            inputs = processor(
                images=images,
                text=captions,
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
    print(f"Epoch {epoch+1}, avg validation loss (val2017): {avg_val_loss:.4f}")

    # Save Hugging Face style checkpoint after each epoch
    ckpt_dir = f"checkpoints_onecap/blip_fixed1cap_finetuned_epoch_{epoch+1}"
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Saving checkpoint to {ckpt_dir}")
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)

# After training, switch to eval mode
model.eval()

# 7. Plot and save validation loss curve
os.makedirs("plots", exist_ok=True)
plt.figure()
plt.plot(range(1, num_epochs + 1), val_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Validation loss")
plt.title("Validation loss over epochs (val2017, fixed one caption)")
plt.grid(True)
plt.savefig("plots/val_loss_fixed1cap.png", dpi=200)
plt.close()
print("Saved validation loss plot to plots/val_loss_fixed1cap.png")


# 8. Evaluation on full val2017 with all 5 captions per image

@torch.no_grad()
def compute_embeddings_for_eval(model, processor, dataset, caps_per_img=5):
    """
    Returns:
      image_features: [M, D]
      text_features: [M * caps_per_img, D]
      where M is the number of images after skipping ones with fewer than caps_per_img captions.
    """
    image_features = []
    text_features = []

    for idx in tqdm(range(len(dataset)), desc="Encoding val2017 for retrieval"):
        image, captions = dataset[idx]
        captions = list(captions)[: caps_per_img]
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


image_feats_val, text_feats_val = compute_embeddings_for_eval(
    model, processor, test_dataset, caps_per_img=caps_per_img
)

M = image_feats_val.size(0)
N_txt = text_feats_val.size(0)
print("Val image_features:", image_feats_val.shape)
print("Val text_features:", text_feats_val.shape)

assert N_txt == M * caps_per_img, f"Expected {M * caps_per_img}, got {N_txt}"

similarity = image_feats_val @ text_feats_val.T
similarity_t2i = similarity.T


def recall_at_k_i2t(sim_matrix, k, caps_per_img=5):
    M = sim_matrix.size(0)
    correct = 0
    for i in range(M):
        sims = sim_matrix[i]
        topk_idx = sims.topk(k).indices.tolist()
        true_indices = set(range(caps_per_img * i, caps_per_img * i + caps_per_img))
        if any(idx in true_indices for idx in topk_idx):
            correct += 1
    return correct / M


def recall_at_k_t2i(sim_matrix, k, caps_per_img=5):
    N_txt = sim_matrix.size(0)
    correct = 0
    for caption_idx in range(N_txt):
        sims = sim_matrix[caption_idx]
        topk_idx = sims.topk(k).indices.tolist()
        true_img_idx = caption_idx // caps_per_img
        if true_img_idx in topk_idx:
            correct += 1
    return correct / N_txt


print("\nVal Image-to-Text (I2T) retrieval on full val2017 (fixed one caption trained):")
for k in [1, 5, 10]:
    r = recall_at_k_i2t(similarity, k, caps_per_img=caps_per_img)
    print(f"R@{k}: {r:.4f}")

print("\nVal Text-to-Image (T2I) retrieval on full val2017 (fixed one caption trained):")
for k in [1, 5, 10]:
    r = recall_at_k_t2i(similarity_t2i, k, caps_per_img=caps_per_img)
    print(f"R@{k}: {r:.4f}")

# 9. Save embeddings for later use
os.makedirs("embeddings", exist_ok=True)
emb_path = "embeddings/fine_tuned_blip_coco_1cap_fixed.pt"
torch.save(
    {
        "image_feats_val": image_feats_val,
        "text_feats_val": text_feats_val,
        "caps_per_img": caps_per_img,
    },
    emb_path,
)
print(f"Saved embeddings to {emb_path}")
