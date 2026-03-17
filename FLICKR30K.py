# eval_flickr30k_retrieval.py

import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Flickr30k
from transformers import BlipProcessor, BlipModel
from tqdm import tqdm


@dataclass
class ModelSpec:
    name: str
    kind: str  # "hf" or "local"
    path: str


def collate_fn_flickr(batch):
    images, captions_list = zip(*batch)
    return list(images), [list(c) for c in captions_list]


@torch.no_grad()
def compute_embeddings_flickr30k(
    model: BlipModel,
    processor: BlipProcessor,
    dataset: Flickr30k,
    device: str,
    caps_per_img: int = 5,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_flickr,
    )

    image_features = []
    text_features = []

    for images, captions_list in tqdm(loader, desc="Encoding Flickr30k"):
        for img, caps in zip(images, captions_list):
            caps = list(caps)[:caps_per_img]
            if len(caps) < caps_per_img:
                continue

            img_inputs = processor(images=img, return_tensors="pt").to(device)
            img_emb = model.get_image_features(**img_inputs)[0]
            image_features.append(img_emb.cpu())

            txt_inputs = processor(text=caps, return_tensors="pt", padding=True).to(device)
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


def recall_at_k_i2t_multi(sim_matrix: torch.Tensor, k: int, caps_per_img: int = 5) -> float:
    m = sim_matrix.size(0)
    correct = 0
    for i in range(m):
        topk_idx = sim_matrix[i].topk(k).indices.tolist()
        true_set = set(range(caps_per_img * i, caps_per_img * i + caps_per_img))
        if any(j in true_set for j in topk_idx):
            correct += 1
    return correct / m


def recall_at_k_t2i_multi(sim_matrix: torch.Tensor, k: int, caps_per_img: int = 5) -> float:
    n_txt = sim_matrix.size(0)
    correct = 0
    for t in range(n_txt):
        topk_idx = sim_matrix[t].topk(k).indices.tolist()
        true_img = t // caps_per_img
        if true_img in topk_idx:
            correct += 1
    return correct / n_txt


def load_blip(model_spec: ModelSpec, device: str) -> Tuple[BlipModel, BlipProcessor]:
    processor = BlipProcessor.from_pretrained(model_spec.path)
    model = BlipModel.from_pretrained(model_spec.path).to(device)
    model.eval()
    return model, processor


def eval_one_model(
    model_spec: ModelSpec,
    dataset: Flickr30k,
    device: str,
    caps_per_img: int,
    batch_size: int,
) -> Dict[str, float]:
    print(f"\nEvaluating: {model_spec.name}")

    model, processor = load_blip(model_spec, device=device)

    img_feats, txt_feats = compute_embeddings_flickr30k(
        model=model,
        processor=processor,
        dataset=dataset,
        device=device,
        caps_per_img=caps_per_img,
        batch_size=batch_size,
    )

    m = img_feats.size(0)
    n_txt = txt_feats.size(0)
    expected = m * caps_per_img
    if n_txt != expected:
        raise RuntimeError(f"Caption count mismatch: expected {expected}, got {n_txt}")

    sim_i2t = img_feats @ txt_feats.T
    sim_t2i = sim_i2t.T

    out = {}
    for k in [1, 5, 10]:
        out[f"I2T_R@{k}"] = recall_at_k_i2t_multi(sim_i2t, k, caps_per_img=caps_per_img)
        out[f"T2I_R@{k}"] = recall_at_k_t2i_multi(sim_t2i, k, caps_per_img=caps_per_img)

    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Edit these two paths for your local Flickr30k setup
    flickr_root = "data/flickr30k/flickr30k-images"
    flickr_ann = "data/flickr30k/results_20130124.token"

    caps_per_img = 5
    batch_size = 32

    dataset = Flickr30k(root=flickr_root, ann_file=flickr_ann)
    print("Flickr30k images:", len(dataset))

    models: List[ModelSpec] = [
        ModelSpec(name="BLIP (orig)", kind="hf", path="Salesforce/blip-itm-base-coco"),
        ModelSpec(name="BLIP + 1 cap (rand)", kind="local", path="/home/mahyarghazanfari/workspace/Project/checkpoints/blip_finetuned_epoch_20"),
        ModelSpec(name="BLIP + 1 cap (fixed)", kind="local", path="/home/mahyarghazanfari/workspace/Project/checkpoints_onecap/blip_fixed1cap_finetuned_epoch_10"),
        ModelSpec(name="BLIP + multi caps", kind="local", path="/home/mahyarghazanfari/workspace/Project/checkpoints_multipositive/blip_multipos_finetuned_epoch_20"),
        ModelSpec(name="BLIP + paras", kind="local", path="/home/mahyarghazanfari/workspace/Project/checkpoints_paragraphs/blip_paragraph_finetuned_epoch_10"),
        ModelSpec(name="BLIP + caps+paras", kind="local", path="/home/mahyarghazanfari/workspace/Project/checkpoints_combined/blip_combined_caps_para_epoch_10"),
    ]

    all_results: Dict[str, Dict[str, float]] = {}
    for ms in models:
        metrics = eval_one_model(
            model_spec=ms,
            dataset=dataset,
            device=device,
            caps_per_img=caps_per_img,
            batch_size=batch_size,
        )
        all_results[ms.name] = metrics

    print("\nSummary (percent):")
    for name, m in all_results.items():
        print(f"\n{name}")
        for k in [1, 5, 10]:
            print(
                f"  I2T R@{k}: {100.0*m[f'I2T_R@{k}']:.1f}   "
                f"T2I R@{k}: {100.0*m[f'T2I_R@{k}']:.1f}"
            )


if __name__ == "__main__":
    main()
