import json
import os
from typing import Dict, Any, Tuple, Optional


TRAIN_JSON = "/home/mahyarghazanfari/workspace/Project/Dataset/generated_coco_train_qwen_llama_with_spice.json"
VAL_JSON   = "/home/mahyarghazanfari/workspace/Project/Dataset/generated_coco_validation_qwen_llama_with_spice.json"


def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def compute_stats(data) -> Dict[str, Any]:
    """
    data: list of dicts, each with keys
      - "paragraph" (may be None or empty)
      - "llama_score" (may be None or missing)
      - "spice" (may be None or missing)
    """
    total = len(data)
    num_with_paragraph = 0

    llama_scores = []
    spice_scores = []

    for item in data:
        paragraph = item.get("paragraph", None)
        has_para = paragraph is not None and str(paragraph).strip() != ""
        if has_para:
            num_with_paragraph += 1

            llama = item.get("llama_score", None)
            if isinstance(llama, (int, float)):
                llama_scores.append(float(llama))

            spice = item.get("spice", None)
            if isinstance(spice, (int, float)):
                spice_scores.append(float(spice))

    num_with_spice = len(spice_scores)
    avg_llama = sum(llama_scores) / len(llama_scores) if llama_scores else None
    avg_spice = sum(spice_scores) / len(spice_scores) if spice_scores else None

    return {
        "total": total,
        "num_with_paragraph": num_with_paragraph,
        "num_with_spice": num_with_spice,
        "avg_llama": avg_llama,
        "avg_spice": avg_spice,
    }


def format_float(x: Optional[float], decimals: int = 3) -> str:
    if x is None:
        return "-"
    return f"{x:.{decimals}f}"


def make_latex_table(train_stats: Dict[str, Any],
                     val_stats: Dict[str, Any]) -> str:
    """
    Returns a LaTeX table string (column width) summarizing train and val stats.
    """
    train_total = train_stats["total"]
    train_para = train_stats["num_with_paragraph"]
    train_spice_n = train_stats["num_with_spice"]
    train_llama = format_float(train_stats["avg_llama"], 2)
    train_spice = format_float(train_stats["avg_spice"], 3)

    val_total = val_stats["total"]
    val_para = val_stats["num_with_paragraph"]
    val_spice_n = val_stats["num_with_spice"]
    val_llama = format_float(val_stats["avg_llama"], 2)
    val_spice = format_float(val_stats["avg_spice"], 3)

    table = r"""
\begin{table}[t]
\centering
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}
\toprule
Split & Total & With paragraph & With SPICE & Avg.\ Llama & Avg.\ SPICE \\
\midrule
Train & %d & %d & %d & %s & %s \\
Val   & %d & %d & %d & %s & %s \\
\bottomrule
\end{tabular}
\caption{Statistics of the generated paragraph dataset on MS COCO train and validation splits. We report the total number of entries, how many retain a non empty paragraph after filtering, how many have a valid SPICE score, and the average Llama and SPICE scores.}
\label{tab:paragraph_dataset_stats}
\end{table}
""" % (
        train_total, train_para, train_spice_n, train_llama, train_spice,
        val_total, val_para, val_spice_n, val_llama, val_spice,
    )

    # Strip leading newlines for cleaner printing
    return table.lstrip()


def main():
    print("Loading JSON files...")
    train_data = load_json(TRAIN_JSON)
    val_data = load_json(VAL_JSON)

    print("Computing stats for train...")
    train_stats = compute_stats(train_data)

    print("Computing stats for val...")
    val_stats = compute_stats(val_data)

    print("Train stats:", train_stats)
    print("Val stats:", val_stats)

    latex_table = make_latex_table(train_stats, val_stats)

    # Print to console so you can copy paste into your paper
    print("\nLaTeX table:\n")
    print(latex_table)

    # Optionally also save it to a .tex file
    out_tex = "paragraph_dataset_stats_table.tex"
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"\nSaved LaTeX table to {out_tex}")


if __name__ == "__main__":
    main()
