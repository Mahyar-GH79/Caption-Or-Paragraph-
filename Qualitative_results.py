# import os
# import json
# import random
# from typing import List, Dict, Any, Tuple

# import torch
# from torch import nn
# from torchvision.datasets import CocoCaptions
# from PIL import Image
# import matplotlib.pyplot as plt
# from transformers import BlipProcessor, BlipModel
# from tqdm import tqdm

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using device:", device)

# # =========================
# # Paths and config
# # =========================
# VAL_ROOT = "data/coco/validation/val2017"
# VAL_ANN = "data/coco/annotations/captions_val2017.json"

# VAL_PARAGRAPH_JSON = "/home/mahyarghazanfari/workspace/Project/Dataset/generated_coco_validation_qwen_llama_with_spice.json"

# # Joint model (trained on caps+paras)
# JOINT_CKPT_DIR = "/home/mahyarghazanfari/workspace/Project/checkpoints_combined/blip_combined_caps_para_epoch_10"  # change if needed

# NUM_EXAMPLES = 5  # number of rows in the qualitative figure
# TOP_K = 5         # number of retrieved images per query
# OUT_FIG_CAPTION = "figures/qualitative_caption_retrieval.png"
# OUT_FIG_PARAGRAPH = "figures/qualitative_paragraph_retrieval.png"

# os.makedirs(os.path.dirname(OUT_FIG_CAPTION), exist_ok=True)

# # =========================
# # Helper: get image path from CocoCaptions
# # =========================
# def get_image_path(dataset: CocoCaptions, idx: int) -> Tuple[str, int]:
#     img_id = dataset.ids[idx]
#     img_info = dataset.coco.loadImgs(img_id)[0]
#     file_name = img_info["file_name"]
#     img_path = os.path.join(dataset.root, file_name)
#     return img_path, img_id


# # =========================
# # Load val COCO and paragraph JSON
# # =========================
# val_coco = CocoCaptions(root=VAL_ROOT, annFile=VAL_ANN, transform=None)
# num_val_images = len(val_coco)
# print("Number of val images:", num_val_images)

# with open(VAL_PARAGRAPH_JSON, "r", encoding="utf-8") as f:
#     val_para_data = json.load(f)

# # Build mapping: image_id -> paragraph
# imgid_to_paragraph: Dict[int, str] = {}
# for item in val_para_data:
#     pid = item.get("image_id", None)
#     para = item.get("paragraph", None)
#     if pid is None:
#         continue
#     if para is None or not str(para).strip():
#         continue
#     imgid_to_paragraph[int(pid)] = para.strip()

# print("Paragraphs available for", len(imgid_to_paragraph), "val images.")

# # =========================
# # Load joint model and processor
# # =========================
# print("Loading joint BLIP model from", JOINT_CKPT_DIR)
# processor = BlipProcessor.from_pretrained(JOINT_CKPT_DIR)
# model = BlipModel.from_pretrained(JOINT_CKPT_DIR).to(device)
# model.eval()

# # =========================
# # Precompute image embeddings for all val images
# # =========================
# @torch.no_grad()
# def compute_val_image_embeddings(
#     model: BlipModel,
#     processor: BlipProcessor,
#     dataset: CocoCaptions,
# ) -> Tuple[torch.Tensor, List[str], List[int]]:
#     """
#     Returns:
#       image_feats: [N, D] tensor of normalized embeddings
#       img_paths:  list of file paths for each embedding
#       img_ids:    list of COCO image ids in same order
#     """
#     image_feats = []
#     img_paths = []
#     img_ids = []

#     for idx in tqdm(range(len(dataset)), desc="Computing val image embeddings"):
#         img_path, img_id = get_image_path(dataset, idx)
#         image = Image.open(img_path).convert("RGB")

#         inputs = processor(images=image, return_tensors="pt").to(device)
#         emb = model.get_image_features(**inputs)[0]  # [D]
#         emb = emb / emb.norm(dim=-1, keepdim=True)

#         image_feats.append(emb.cpu())
#         img_paths.append(img_path)
#         img_ids.append(int(img_id))

#     image_feats = torch.stack(image_feats, dim=0)  # [N, D]
#     return image_feats, img_paths, img_ids


# image_feats_val, val_img_paths, val_img_ids = compute_val_image_embeddings(
#     model, processor, val_coco
# )
# print("Val image_feats shape:", image_feats_val.shape)


# # =========================
# # Sample examples that have both captions and paragraphs
# # =========================
# def sample_val_examples(
#     dataset: CocoCaptions,
#     imgid_to_paragraph: Dict[int, str],
#     num_examples: int,
# ) -> List[int]:
#     """
#     Returns a list of dataset indices that have at least one caption
#     and an associated paragraph in the JSON.
#     """
#     valid_indices = []
#     for idx in range(len(dataset)):
#         image, captions = dataset[idx]
#         captions = list(captions)
#         if len(captions) == 0:
#             continue
#         _, img_id = get_image_path(dataset, idx)
#         if int(img_id) in imgid_to_paragraph:
#             valid_indices.append(idx)

#     print("Total val indices with captions and paragraph:", len(valid_indices))

#     if len(valid_indices) < num_examples:
#         raise ValueError("Not enough valid examples to sample from.")

#     random.seed(42)
#     sampled = random.sample(valid_indices, num_examples)
#     return sampled


# sampled_indices = sample_val_examples(val_coco, imgid_to_paragraph, NUM_EXAMPLES)
# print("Sampled indices:", sampled_indices)


# # =========================
# # Text embedding helper
# # =========================
# @torch.no_grad()
# def encode_text(query: str) -> torch.Tensor:
#     inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
#     feats = model.get_text_features(
#         input_ids=inputs.input_ids,
#         attention_mask=inputs.attention_mask,
#     )[0]  # [D]
#     feats = feats / feats.norm(dim=-1, keepdim=True)
#     return feats.cpu()


# # =========================
# # Retrieval helper
# # =========================
# def retrieve_topk_images(
#     text_feat: torch.Tensor,
#     image_feats: torch.Tensor,
#     k: int,
# ) -> List[int]:
#     """
#     text_feat: [D]
#     image_feats: [N, D]
#     Returns the indices of top k most similar images.
#     """
#     sims = text_feat @ image_feats.T  # [N]
#     topk = torch.topk(sims, k=k)
#     return topk.indices.tolist()


# # =========================
# # Qualitative plot helper
# # =========================
# def make_qualitative_figure(
#     dataset: CocoCaptions,
#     image_feats: torch.Tensor,
#     img_paths: List[str],
#     img_ids: List[int],
#     sampled_indices: List[int],
#     imgid_to_paragraph: Dict[int, str],
#     query_type: str,
#     out_path: str,
#     top_k: int = 5,
# ):
#     """
#     query_type: "caption" or "paragraph"
#     Builds a NUM_EXAMPLES x (top_k + 1) grid:
#       columns 0..top_k-1: retrieved images
#       column top_k: ground truth image
#     """
#     assert query_type in ["caption", "paragraph"]
#     num_rows = len(sampled_indices)
#     num_cols = top_k + 1

#     fig, axes = plt.subplots(
#         num_rows,
#         num_cols,
#         figsize=(3 * num_cols, 3 * num_rows),
#     )

#     if num_rows == 1:
#         axes = axes.reshape(1, -1)

#     for row_idx, ds_idx in enumerate(sampled_indices):
#         gt_img_path, gt_img_id = get_image_path(dataset, ds_idx)
#         image, captions = dataset[ds_idx]
#         captions = list(captions)

#         if query_type == "caption":
#             # choose a random caption
#             query_text = random.choice(captions)
#             row_title = f"Cap: {query_text}"
#         else:
#             query_text = imgid_to_paragraph[int(gt_img_id)]
#             row_title = f"Para: {query_text}"

#         # encode query
#         text_feat = encode_text(query_text)
#         # retrieve top k
#         retrieved_indices = retrieve_topk_images(text_feat, image_feats, k=top_k)

#         # Draw retrieved images
#         for col_idx, img_index in enumerate(retrieved_indices):
#             ax = axes[row_idx, col_idx]
#             img = Image.open(img_paths[img_index]).convert("RGB")
#             ax.imshow(img)
#             ax.axis("off")
#             # Mark if this retrieved image is the ground truth
#             if img_ids[img_index] == int(gt_img_id):
#                 ax.set_title(f"Rank {col_idx+1} (GT)", fontsize=8, color="green")
#             else:
#                 ax.set_title(f"Rank {col_idx+1}", fontsize=8)

#         # Draw ground truth in last column
#         ax_gt = axes[row_idx, top_k]
#         gt_img = Image.open(gt_img_path).convert("RGB")
#         ax_gt.imshow(gt_img)
#         ax_gt.axis("off")
#         ax_gt.set_title("Ground truth", fontsize=8, color="blue")

#         # Put a truncated text above the row
#         # Truncate to keep figure readable
#         max_chars = 80
#         short_query = (query_text[:max_chars] + "…") if len(query_text) > max_chars else query_text
#         fig.text(
#             0.5,
#             1.0 - (row_idx + 0.5) / num_rows,
#             short_query,
#             ha="center",
#             va="center",
#             fontsize=9,
#         )

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.92)
#     title = "Caption based retrieval" if query_type == "caption" else "Paragraph based retrieval"
#     fig.suptitle(title, fontsize=12)
#     fig.savefig(out_path, dpi=200)
#     plt.close(fig)
#     print(f"Saved qualitative figure to {out_path}")


# # =========================
# # Build caption based figure
# # =========================
# make_qualitative_figure(
#     dataset=val_coco,
#     image_feats=image_feats_val,
#     img_paths=val_img_paths,
#     img_ids=val_img_ids,
#     sampled_indices=sampled_indices,
#     imgid_to_paragraph=imgid_to_paragraph,
#     query_type="caption",
#     out_path=OUT_FIG_CAPTION,
#     top_k=TOP_K,
# )

# # =========================
# # Build paragraph based figure
# # =========================
# make_qualitative_figure(
#     dataset=val_coco,
#     image_feats=image_feats_val,
#     img_paths=val_img_paths,
#     img_ids=val_img_ids,
#     sampled_indices=sampled_indices,
#     imgid_to_paragraph=imgid_to_paragraph,
#     query_type="paragraph",
#     out_path=OUT_FIG_PARAGRAPH,
#     top_k=TOP_K,
# )

