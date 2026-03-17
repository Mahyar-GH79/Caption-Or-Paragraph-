import torch
from torchvision.datasets import CocoCaptions
from transformers import BlipProcessor, BlipModel
from tqdm import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

root = "data/coco/val2017"
ann_file = "data/coco/annotations/captions_val2017.json"

# 1. Load COCO
coco = CocoCaptions(root=root, annFile=ann_file, transform=None)
N_img = len(coco)
print("Number of images:", N_img)

# 2. Load BLIP model
model_name = "Salesforce/blip-itm-base-coco"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipModel.from_pretrained(model_name).to(device)
model.eval()

image_features = []
text_features = []

caps_per_img = 5  # we force exactly 5 captions per image

# 3. Encode images and first 5 captions
with torch.no_grad():
    for idx in tqdm(range(N_img), desc="Encoding"):
        image, captions = coco[idx]          # captions: list of strings
        captions = list(captions)[:caps_per_img]  # take only first 5

        # safety: if for some reason fewer than 5 captions, skip this image
        if len(captions) < caps_per_img:
            # you can also choose to pad/duplicate instead of skipping
            continue

        # Image embedding
        img_inputs = processor(
            images=image,
            return_tensors="pt",
        ).to(device)

        img_emb = model.get_image_features(**img_inputs)  # [1, D]
        img_emb = img_emb[0]                              # [D]
        image_features.append(img_emb.cpu())

        # Text embeddings for 5 captions
        txt_inputs = processor(
            text=captions,
            return_tensors="pt",
            padding=True,
        ).to(device)

        txt_embs = model.get_text_features(**txt_inputs)  # [5, D]
        text_features.append(txt_embs.cpu())

# 4. Stack and normalize
image_features = torch.stack(image_features, dim=0)       # [M, D]
text_features = torch.cat(text_features, dim=0)           # [5M, D]

M = image_features.size(0)  # effective number of images actually used
N_txt = text_features.size(0)

print("Effective number of images (M):", M)
print("image_features shape:", image_features.shape)
print("text_features shape:", text_features.shape)

# sanity check: should be exactly 5 captions per used image
assert N_txt == M * caps_per_img, f"Expected {M * caps_per_img} captions, got {N_txt}"

image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# 5. Save embeddings
os.makedirs("embeddings", exist_ok=True)
save_path = os.path.join("embeddings", "blip_coco_5caps_clean.pt")
torch.save(
    {
        "image_features": image_features,
        "text_features": text_features,
    },
    save_path,
)
print("Saved embeddings to", save_path)

# 6. Similarity matrix: [M, 5M]
similarity = image_features @ text_features.T  # [M, 5M]

# 7. I2T: Image -> Text
def recall_at_k_i2t(sim_matrix, k, caps_per_img=5):
    M = sim_matrix.size(0)
    correct = 0
    for i in range(M):
        sims = sim_matrix[i]                    # [5M]
        topk_idx = sims.topk(k).indices.tolist()

        true_indices = set(range(caps_per_img * i, caps_per_img * i + caps_per_img))

        if any(idx in true_indices for idx in topk_idx):
            correct += 1

    return correct / M

# 8. T2I: Text -> Image
similarity_t2i = similarity.T  # [5M, M]

def recall_at_k_t2i(sim_matrix, k, caps_per_img=5):
    N_txt = sim_matrix.size(0)
    correct = 0
    for caption_idx in range(N_txt):
        sims = sim_matrix[caption_idx]          # [M]
        topk_idx = sims.topk(k).indices.tolist()

        true_img_idx = caption_idx // caps_per_img

        if true_img_idx in topk_idx:
            correct += 1

    return correct / N_txt

print("\nImage-to-Text (I2T) retrieval:")
for k in [1, 5, 10]:
    r = recall_at_k_i2t(similarity, k, caps_per_img=caps_per_img)
    print(f"R@{k}: {r:.4f}")

print("\nText-to-Image (T2I) retrieval:")
for k in [1, 5, 10]:
    r = recall_at_k_t2i(similarity_t2i, k, caps_per_img=caps_per_img)
    print(f"R@{k}: {r:.4f}")
