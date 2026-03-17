import os
import json
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoCaptions
from PIL import Image
from transformers import BlipProcessor, BlipModel
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# COCO val paths
VAL_ROOT = "data/coco/validation/val2017"
VAL_ANN  = "data/coco/annotations/captions_val2017.json"

# Paragraph JSON for val (Qwen + Llama + SPICE)
VAL_JSON = "/home/mahyarghazanfari/workspace/Project/Dataset/generated_coco_validation_qwen_llama_with_spice.json"

# Paragraph filters (optional; set to None for no filtering)
MIN_LLAMA_SCORE = None
MIN_SPICE       = None

# Captions per image for COCO eval
CAPS_PER_IMG = 5
BATCH_SIZE_CAPTIONS   = 32
BATCH_SIZE_PARAGRAPHS = 32

# Model identifiers / checkpoint directories
ORIG_MODEL_ID     = "Salesforce/blip-itm-base-coco"

RANDCAP_CKPT_DIR  = "/home/mahyarghazanfari/workspace/Project/checkpoints/blip_finetuned_epoch_20"
FIXED1CAP_CKPT_DIR = "/home/mahyarghazanfari/workspace/Project/checkpoints_onecap/blip_fixed1cap_finetuned_epoch_10"
MULTIPOS_CKPT_DIR = "/home/mahyarghazanfari/workspace/Project/checkpoints_multipositive/blip_multipos_finetuned_epoch_20"
PARA_CKPT_DIR     = "/home/mahyarghazanfari/workspace/Project/checkpoints_paragraphs/blip_paragraph_finetuned_epoch_10"
COMBINED_CKPT_DIR = "/home/mahyarghazanfari/workspace/Project/checkpoints_combined/blip_combined_caps_para_epoch_10"

# Names to show in LaTeX table (rows in order)
MODEL_CONFIGS = [
    {"name": "BLIP (orig)",        "type": "hf",    "path": ORIG_MODEL_ID},
    {"name": "BLIP + 1 cap (rand)","type": "local", "path": RANDCAP_CKPT_DIR},
    {"name": "BLIP + 1 cap (fixed)","type": "local","path": FIXED1CAP_CKPT_DIR},
    {"name": "BLIP + multi caps",  "type": "local", "path": MULTIPOS_CKPT_DIR},
    {"name": "BLIP + paras",       "type": "local", "path": PARA_CKPT_DIR},
    {"name": "BLIP + caps+paras",  "type": "local", "path": COMBINED_CKPT_DIR},
]

# Metric keys in the LaTeX table order
METRIC_KEYS = [
    "cap_I2T_R@1", "cap_I2T_R@5", "cap_I2T_R@10",
    "cap_T2I_R@1", "cap_T2I_R@5", "cap_T2I_R@10",
    "para_I2T_R@1","para_I2T_R@5","para_I2T_R@10",
    "para_T2I_R@1","para_T2I_R@5","para_T2I_R@10",
]


# ============================================================
# DATASETS
# ============================================================

class CocoAllCaptionsDataset(Dataset):
    """
    For caption retrieval:
      __getitem__ -> (image, [up to caps_per_img captions])
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


class ParagraphValDataset(Dataset):
    """
    Paragraph-only dataset for retrieval.
    Returns: (image, paragraph)
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
                {"image_path": img_path, "paragraph": paragraph.strip()}
            )

        self.entries = entries
        print(
            f"[{split_name}] Loaded {len(self.entries)} image-paragraph pairs "
            f"(filters: llama>={min_llama_score}, spice>={min_spice})"
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        return image, item["paragraph"]


def collate_fn_caps(batch):
    images, captions_list = zip(*batch)
    return list(images), [list(caps) for caps in captions_list]


def collate_fn_paras(batch):
    images, paragraphs = zip(*batch)
    return list(images), list(paragraphs)


# ============================================================
# EVAL HELPERS
# ============================================================

@torch.no_grad()
def compute_embeddings_for_eval_captions(
    model: BlipModel,
    processor: BlipProcessor,
    dataset: CocoAllCaptionsDataset,
    caps_per_img: int = 5,
    batch_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_caps,
    )

    image_features = []
    text_features = []

    for images, captions_list in tqdm(loader, desc="Encoding val captions"):
        for image, caps in zip(images, captions_list):
            caps = caps[:caps_per_img]
            if len(caps) < caps_per_img:
                continue

            img_inputs = processor(images=image, return_tensors="pt").to(device)
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


@torch.no_grad()
def compute_embeddings_for_eval_paragraphs(
    model: BlipModel,
    processor: BlipProcessor,
    dataset: ParagraphValDataset,
    batch_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_paras,
    )

    image_features = []
    text_features = []

    for images, paragraphs in tqdm(loader, desc="Encoding val paragraphs"):
        img_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        txt_inputs = processor(text=list(paragraphs), return_tensors="pt", padding=True).to(device)

        img_embs = model.get_image_features(pixel_values=img_inputs.pixel_values)
        txt_embs = model.get_text_features(
            input_ids=txt_inputs.input_ids,
            attention_mask=txt_inputs.attention_mask,
        )

        image_features.append(img_embs.cpu())
        text_features.append(txt_embs.cpu())

    image_features = torch.cat(image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return image_features, text_features


def recall_at_k_i2t_multi(sim_matrix: torch.Tensor, k: int, caps_per_img: int = 5) -> float:
    M = sim_matrix.size(0)
    correct = 0
    for i in range(M):
        topk_idx = sim_matrix[i].topk(k).indices.tolist()
        true_indices = set(range(caps_per_img * i, caps_per_img * i + caps_per_img))
        if any(idx in true_indices for idx in topk_idx):
            correct += 1
    return correct / M


def recall_at_k_t2i_multi(sim_matrix: torch.Tensor, k: int, caps_per_img: int = 5) -> float:
    N_txt = sim_matrix.size(0)
    correct = 0
    for caption_idx in range(N_txt):
        topk_idx = sim_matrix[caption_idx].topk(k).indices.tolist()
        true_img_idx = caption_idx // caps_per_img
        if true_img_idx in topk_idx:
            correct += 1
    return correct / N_txt


def recall_at_k_one_positive(sim_matrix: torch.Tensor, k: int) -> float:
    M = sim_matrix.size(0)
    correct = 0
    for i in range(M):
        topk_idx = sim_matrix[i].topk(k).indices.tolist()
        if i in topk_idx:
            correct += 1
    return correct / M


# ============================================================
# MAIN EVAL ROUTINE
# ============================================================

def eval_model(
    model_name: str,
    load_type: str,
    path: str,
    val_captions_dataset: CocoAllCaptionsDataset,
    val_paragraph_dataset: ParagraphValDataset,
    caps_per_img: int = 5,
) -> Dict[str, float]:
    print(f"\n=== Evaluating model: {model_name} ===")

    if load_type == "hf":
        processor = BlipProcessor.from_pretrained(path)
        model = BlipModel.from_pretrained(path).to(device)
    else:
        assert os.path.isdir(path), f"Checkpoint dir not found: {path}"
        processor = BlipProcessor.from_pretrained(path)
        model = BlipModel.from_pretrained(path).to(device)

    model.eval()

    # ---- Captions retrieval ----
    img_caps, txt_caps = compute_embeddings_for_eval_captions(
        model,
        processor,
        val_captions_dataset,
        caps_per_img=caps_per_img,
        batch_size=BATCH_SIZE_CAPTIONS,
    )
    sim_caps = img_caps @ txt_caps.T
    sim_caps_t2i = sim_caps.T

    metrics: Dict[str, float] = {}
    for k in [1, 5, 10]:
        metrics[f"cap_I2T_R@{k}"] = recall_at_k_i2t_multi(sim_caps, k, caps_per_img=caps_per_img)
        metrics[f"cap_T2I_R@{k}"] = recall_at_k_t2i_multi(sim_caps_t2i, k, caps_per_img=caps_per_img)

    # ---- Paragraph retrieval ----
    img_para, txt_para = compute_embeddings_for_eval_paragraphs(
        model,
        processor,
        val_paragraph_dataset,
        batch_size=BATCH_SIZE_PARAGRAPHS,
    )
    sim_para = img_para @ txt_para.T
    sim_para_t2i = sim_para.T

    for k in [1, 5, 10]:
        metrics[f"para_I2T_R@{k}"] = recall_at_k_one_positive(sim_para, k)
        metrics[f"para_T2I_R@{k}"] = recall_at_k_one_positive(sim_para_t2i, k)

    return metrics


# ============================================================
# LaTeX TABLE WITH BEST / SECOND BEST HIGHLIGHTING
# ============================================================

def _rank_best_second(values: List[float]) -> Tuple[int, int]:
    """
    Returns indices (best_idx, second_best_idx) for a list of floats.
    Ties:
      - best: first occurrence of max
      - second: first occurrence of next lower value
      - if all values equal, second_best_idx = -1
    """
    if len(values) == 0:
        return -1, -1
    # unique sorted descending
    uniq = sorted(set(values), reverse=True)
    best_val = uniq[0]
    best_idx = values.index(best_val)
    if len(uniq) == 1:
        return best_idx, -1
    second_val = uniq[1]
    second_idx = values.index(second_val)
    return best_idx, second_idx


def make_latex_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    results: model_name -> metrics dict (fractions in [0,1])
    Produces:
      - Best per column: \\textbf{..}
      - Second best per column: \\uline{..}
    Requires LaTeX: \\usepackage[normalem]{ulem}
    """
    # Build matrix: rows in MODEL_CONFIGS order, cols in METRIC_KEYS order
    row_names = [cfg["name"] for cfg in MODEL_CONFIGS]
    mat = []
    for name in row_names:
        m = results[name]
        mat.append([float(m[k]) for k in METRIC_KEYS])

    # Determine best and second best indices per column
    best_second_per_col: List[Tuple[int, int]] = []
    for col in range(len(METRIC_KEYS)):
        col_vals = [mat[r][col] for r in range(len(row_names))]
        best_idx, second_idx = _rank_best_second(col_vals)
        best_second_per_col.append((best_idx, second_idx))

    def fmt_cell(value: float, is_best: bool, is_second: bool) -> str:
        s = f"{100.0 * value:.1f}"
        if is_best:
            return r"\textbf{" + s + "}"
        if is_second:
            return r"\uline{" + s + "}"
        return s

    header = r"""
\begin{table}[t]
\centering
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccccccccc}
\toprule
& \multicolumn{3}{c}{Cap I2T} & \multicolumn{3}{c}{Cap T2I} & \multicolumn{3}{c}{Para I2T} & \multicolumn{3}{c}{Para T2I} \\
Model & R@1 & R@5 & R@10 & R@1 & R@5 & R@10 & R@1 & R@5 & R@10 & R@1 & R@5 & R@10 \\
\midrule
""".strip("\n")

    lines = [header]

    for r, name in enumerate(row_names):
        cells = []
        for c, key in enumerate(METRIC_KEYS):
            best_r, second_r = best_second_per_col[c]
            cells.append(fmt_cell(mat[r][c], is_best=(r == best_r), is_second=(r == second_r)))

        line = (
            f"{name} & "
            + " & ".join(cells)
            + r" \\"
        )
        lines.append(line)

    footer = r"""
\bottomrule
\end{tabular}
\caption{Cross-modal retrieval performance on MS-COCO validation for captions and generated paragraphs. We report Recall@K (\%, higher is better) for image-to-text (I2T) and text-to-image (T2I). Best in \textbf{bold} and second best \uline{underlined}.}
\label{tab:blip_paragraph_caption_retrieval}
\end{table}
""".strip("\n")

    lines.append(footer)
    return "\n".join(lines)


# ============================================================
# RUN EVERYTHING
# ============================================================

def main():
    val_coco = CocoCaptions(root=VAL_ROOT, annFile=VAL_ANN, transform=None)
    print("Number of val images (COCO):", len(val_coco))

    val_captions_dataset = CocoAllCaptionsDataset(val_coco, caps_per_img=CAPS_PER_IMG)

    val_paragraph_dataset = ParagraphValDataset(
        VAL_JSON,
        split_name="val_paragraphs",
        min_llama_score=MIN_LLAMA_SCORE,
        min_spice=MIN_SPICE,
    )

    all_results: Dict[str, Dict[str, float]] = {}

    for cfg in MODEL_CONFIGS:
        name = cfg["name"]
        metrics = eval_model(
            model_name=name,
            load_type=cfg["type"],
            path=cfg["path"],
            val_captions_dataset=val_captions_dataset,
            val_paragraph_dataset=val_paragraph_dataset,
            caps_per_img=CAPS_PER_IMG,
        )
        all_results[name] = metrics

    print("\n==== Raw metrics (fraction) ====")
    for name, m in all_results.items():
        print(f"\n{name}:")
        for k in [1, 5, 10]:
            print(
                f"  Cap I2T R@{k}: {m[f'cap_I2T_R@{k}']:.4f}, "
                f"Cap T2I R@{k}: {m[f'cap_T2I_R@{k}']:.4f}, "
                f"Para I2T R@{k}: {m[f'para_I2T_R@{k}']:.4f}, "
                f"Para T2I R@{k}: {m[f'para_T2I_R@{k}']:.4f}"
            )

    latex_table = make_latex_table(all_results)
    print("\n\n==== LaTeX table (copy into CVPR paper) ====\n")
    print(latex_table)

    os.makedirs("tables", exist_ok=True)
    out_path = "tables/retrieval_results_blip_paragraphs_captions.tex"
    with open(out_path, "w") as f:
        f.write(latex_table)
    print(f"\nSaved LaTeX table to {out_path}")
    print("\nReminder: add this to your LaTeX preamble: \\usepackage[normalem]{ulem}")

if __name__ == "__main__":
    main()
