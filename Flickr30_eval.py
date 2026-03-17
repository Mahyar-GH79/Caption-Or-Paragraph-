import os
import csv
import json
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipModel

# =========================
# Config you must set
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# Point these to YOUR paths
FLICKR_IMAGES_DIR = "/home/mahyarghazanfari/workspace/Project/data/flickr30k/flickr30k_images"   
CAPTIONS_CSV_PATH = "/home/mahyarghazanfari/workspace/Project/data/flickr30k/captions.txt"       

CAPS_PER_IMG = 5
BATCH_SIZE_IMAGES = 32          # batches of images (each image has 5 captions)
NUM_WORKERS = 4

# Your checkpoints
ORIG_MODEL_ID = "Salesforce/blip-itm-base-coco"

RANDCAP_CKPT_DIR   = "/home/mahyarghazanfari/workspace/Project/checkpoints/blip_finetuned_epoch_20"
FIXED1CAP_CKPT_DIR = "/home/mahyarghazanfari/workspace/Project/checkpoints_onecap/blip_fixed1cap_finetuned_epoch_10"
MULTIPOS_CKPT_DIR  = "/home/mahyarghazanfari/workspace/Project/checkpoints_multipositive/blip_multipos_finetuned_epoch_20"
PARA_CKPT_DIR      = "/home/mahyarghazanfari/workspace/Project/checkpoints_paragraphs/blip_paragraph_finetuned_epoch_10"
COMBINED_CKPT_DIR  = "/home/mahyarghazanfari/workspace/Project/checkpoints_combined/blip_combined_caps_para_epoch_10"

MODEL_CONFIGS = [
    {"name": "BLIP (orig)",         "type": "hf",    "path": ORIG_MODEL_ID},
    {"name": "BLIP + 1 cap (rand)", "type": "local", "path": RANDCAP_CKPT_DIR},
    {"name": "BLIP + 1 cap (fixed)","type": "local", "path": FIXED1CAP_CKPT_DIR},
    {"name": "BLIP + multi caps",   "type": "local", "path": MULTIPOS_CKPT_DIR},
    {"name": "BLIP + paras",        "type": "local", "path": PARA_CKPT_DIR},
    {"name": "BLIP + caps+paras",   "type": "local", "path": COMBINED_CKPT_DIR},
]

# =========================
# Load captions CSV
# =========================
def load_captions_csv(path: str, caps_per_img: int = 5) -> Dict[str, List[str]]:
    """
    Returns: image_name -> list of captions (kept in comment_number order if possible)
    Keeps only images with >= caps_per_img captions.
    """
    assert os.path.exists(path), f"Captions file not found: {path}"

    tmp: Dict[str, List[Tuple[int, str]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image_name", "comment_number", "comment"}
        if set(reader.fieldnames or []) != required:
            raise ValueError(f"Unexpected header: {reader.fieldnames}. Expected {required}")

        for row in reader:
            img = row["image_name"].strip()
            cap = row["comment"].strip()
            if not img or not cap:
                continue
            try:
                idx = int(row["comment_number"])
            except Exception:
                idx = 0
            tmp.setdefault(img, []).append((idx, cap))

    img_to_caps: Dict[str, List[str]] = {}
    for img, pairs in tmp.items():
        pairs = sorted(pairs, key=lambda x: x[0])  # order by comment_number
        caps = [c for _, c in pairs]
        if len(caps) >= caps_per_img:
            img_to_caps[img] = caps[:caps_per_img]

    print(f"Captions loaded for {len(img_to_caps)} images with >= {caps_per_img} captions.")
    return img_to_caps

# =========================
# Dataset
# =========================
class FlickrCSVCaptionDataset(Dataset):
    """
    Each item: (PIL_image, captions_list_of_len_K)
    """
    def __init__(self, images_dir: str, img_to_caps: Dict[str, List[str]], caps_per_img: int = 5):
        self.items: List[Tuple[str, List[str]]] = []
        self.caps_per_img = caps_per_img

        for img_name, caps in img_to_caps.items():
            img_path = os.path.join(images_dir, img_name)
            if os.path.exists(img_path):
                if len(caps) >= caps_per_img:
                    self.items.append((img_path, caps[:caps_per_img]))

        if len(self.items) == 0:
            raise RuntimeError(
                "No usable items found. Check FLICKR_IMAGES_DIR and that filenames match captions.txt."
            )

        print(f"Usable image-caption pairs: {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, caps = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        return image, caps

def collate_fn(batch):
    images, caps_list = zip(*batch)
    return list(images), list(caps_list)

# =========================
# Metrics (multi-positive)
# =========================
def recall_at_k_i2t_multi(sim: torch.Tensor, k: int, caps_per_img: int) -> float:
    """
    sim: [M, M*caps_per_img]
    positives for image i: caption indices [caps_per_img*i .. caps_per_img*i+caps_per_img-1]
    """
    m = sim.size(0)
    correct = 0
    for i in range(m):
        topk = sim[i].topk(k).indices.tolist()
        true = set(range(caps_per_img * i, caps_per_img * i + caps_per_img))
        if any(t in true for t in topk):
            correct += 1
    return correct / m

def recall_at_k_t2i_multi(sim_t2i: torch.Tensor, k: int, caps_per_img: int) -> float:
    """
    sim_t2i: [M*caps_per_img, M]
    positive image for caption t: t // caps_per_img
    """
    n_txt = sim_t2i.size(0)
    correct = 0
    for t in range(n_txt):
        topk = sim_t2i[t].topk(k).indices.tolist()
        true_img = t // caps_per_img
        if true_img in topk:
            correct += 1
    return correct / n_txt

# =========================
# Embedding computation (batched)
# =========================
@torch.no_grad()
def compute_embeddings(
    model: BlipModel,
    processor: BlipProcessor,
    dataset: Dataset,
    batch_size_images: int,
    caps_per_img: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      img_feats: [M, D]
      txt_feats: [M*caps_per_img, D]
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size_images,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    model.eval()
    img_feats_all = []
    txt_feats_all = []

    for images, caps_list in tqdm(loader, desc="Encoding Flickr30k"):
        # images: list[PIL] length B
        # caps_list: list[list[str]] length B, each list length K

        # 1) encode images in a batch
        img_inputs = processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
        img_embs = model.get_image_features(pixel_values=img_inputs.pixel_values)  # [B, D]
        img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
        img_feats_all.append(img_embs.cpu())

        # 2) encode texts in a batch (flatten B*K)
        flat_caps: List[str] = []
        for caps in caps_list:
            caps = caps[:caps_per_img]
            if len(caps) != caps_per_img:
                # should not happen since dataset filtered, but safety
                caps = (caps + [""] * caps_per_img)[:caps_per_img]
            flat_caps.extend(caps)

        txt_inputs = processor(text=flat_caps, return_tensors="pt", padding=True).to(DEVICE)
        txt_embs = model.get_text_features(
            input_ids=txt_inputs.input_ids,
            attention_mask=txt_inputs.attention_mask,
        )  # [B*K, D]
        txt_embs = txt_embs / txt_embs.norm(dim=-1, keepdim=True)
        txt_feats_all.append(txt_embs.cpu())

    img_feats = torch.cat(img_feats_all, dim=0)
    txt_feats = torch.cat(txt_feats_all, dim=0)

    # Sanity: txt should be img * K
    m = img_feats.size(0)
    assert txt_feats.size(0) == m * caps_per_img, f"Expected {m * caps_per_img}, got {txt_feats.size(0)}"

    return img_feats, txt_feats

def load_blip(cfg: Dict) -> Tuple[BlipModel, BlipProcessor]:
    if cfg["type"] == "hf":
        processor = BlipProcessor.from_pretrained(cfg["path"])
        model = BlipModel.from_pretrained(cfg["path"]).to(DEVICE)
    else:
        if not os.path.isdir(cfg["path"]):
            raise FileNotFoundError(f"Checkpoint dir not found: {cfg['path']}")
        processor = BlipProcessor.from_pretrained(cfg["path"])
        model = BlipModel.from_pretrained(cfg["path"]).to(DEVICE)
    return model, processor

# =========================
# Main
# =========================
def main():
    img_to_caps = load_captions_csv(CAPTIONS_CSV_PATH, caps_per_img=CAPS_PER_IMG)
    ds = FlickrCSVCaptionDataset(FLICKR_IMAGES_DIR, img_to_caps, caps_per_img=CAPS_PER_IMG)

    results = {}

    for cfg in MODEL_CONFIGS:
        name = cfg["name"]
        print("\n==============================")
        print("Evaluating:", name)

        model, processor = load_blip(cfg)

        img_feats, txt_feats = compute_embeddings(
            model=model,
            processor=processor,
            dataset=ds,
            batch_size_images=BATCH_SIZE_IMAGES,
            caps_per_img=CAPS_PER_IMG,
        )

        sim = img_feats @ txt_feats.T              # [M, M*K]
        sim_t2i = sim.T                            # [M*K, M]

        metrics = {}
        for k in [1, 5, 10]:
            metrics[f"I2T_R@{k}"] = recall_at_k_i2t_multi(sim, k, CAPS_PER_IMG)
            metrics[f"T2I_R@{k}"] = recall_at_k_t2i_multi(sim_t2i, k, CAPS_PER_IMG)

        results[name] = metrics

        print("Recall results:")
        for k in [1, 5, 10]:
            print(f"  I2T R@{k}: {metrics[f'I2T_R@{k}']:.4f}    T2I R@{k}: {metrics[f'T2I_R@{k}']:.4f}")

    os.makedirs("tables", exist_ok=True)
    out_json = "tables/flickr30k_retrieval_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:", out_json)

    # Small pretty print summary
    print("\n==== Summary (percent) ====")
    for name, m in results.items():
        print(
            f"{name:22s} | "
            f"I2T R@1 {100*m['I2T_R@1']:.1f}  R@5 {100*m['I2T_R@5']:.1f}  R@10 {100*m['I2T_R@10']:.1f} | "
            f"T2I R@1 {100*m['T2I_R@1']:.1f}  R@5 {100*m['T2I_R@5']:.1f}  R@10 {100*m['T2I_R@10']:.1f}"
        )

if __name__ == "__main__":
    main()
