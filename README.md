# Caption or Paragraph? Exploring Image Description Strategies for Vision-Language Models

This repository contains the code for fine-tuning and evaluating vision-language models on image captioning and paragraph-level description tasks. The project investigates whether single captions or detailed paragraphs produce better image-text representations, using models like Qwen and LLaMA on datasets including COCO, Flickr30K, CC3M, and ShareGPT4V. You can find the complete report [here](Caption_Or_Paragraph_Report.pdf).

## Project Structure

```
Caption-Or-Paragraph-/
├── training/                          # Fine-tuning scripts
│   ├── fine_tuning_captions.py        # Fine-tune with single captions
│   ├── fine_tuning_paragraphs.py      # Fine-tune with paragraph descriptions
│   ├── fine_tuning_captionandparagraph.py  # Fine-tune with both captions and paragraphs
│   └── fine_tuning_one_fixed_caption.py    # Fine-tune with one fixed caption per image
│
├── evaluation/                        # Evaluation and analysis scripts
│   ├── baseline.py                    # Baseline model evaluation
│   ├── Final_eval.py                  # Final evaluation pipeline
│   ├── Flickr30_eval.py               # Evaluation on Flickr30K
│   ├── Cross_Validation.py            # Cross-validation experiments
│   └── Qualitative_results.py         # Generate qualitative result samples
│
├── data_processing/                   # Dataset preparation and processing
│   ├── COCO.py                        # COCO dataset processing
│   ├── COCO_train.py                  # COCO training split preparation
│   ├── COCO_train_val.py              # COCO train/val split preparation
│   ├── CC3M.py                        # CC3M dataset processing
│   ├── FLICKR30K.py                   # Flickr30K dataset processing
│   ├── ShareGPT4v.py                  # ShareGPT4V dataset processing
│   └── datasetinfo.py                 # Dataset statistics and information
│
├── plots/                             # Generated plots and figures
├── Dataset/                           # Dataset files (large files excluded)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Mahyar-GH79/Caption-Or-Paragraph-.git
cd Caption-Or-Paragraph-
```

### 2. Create a virtual environment

```bash
python -m venv env
source env/bin/activate  # Linux/Mac
# or
env\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Data Processing

Prepare the datasets before training:

```bash
python data_processing/COCO.py
python data_processing/COCO_train.py
python data_processing/CC3M.py
python data_processing/FLICKR30K.py
python data_processing/ShareGPT4v.py
```

### Training

Fine-tune models with different description strategies:

```bash
# Fine-tune with single captions
python training/fine_tuning_captions.py

# Fine-tune with paragraph descriptions
python training/fine_tuning_paragraphs.py

# Fine-tune with both captions and paragraphs
python training/fine_tuning_captionandparagraph.py
```

### Evaluation

```bash
# Run baseline evaluation
python evaluation/baseline.py

# Run final evaluation
python evaluation/Final_eval.py

# Evaluate on Flickr30K
python evaluation/Flickr30_eval.py

# Cross-validation
python evaluation/Cross_Validation.py

# Generate qualitative results
python evaluation/Qualitative_results.py
```

## Datasets

This project uses the following datasets:

| Dataset | Description |
|---------|-------------|
| **COCO** | Microsoft Common Objects in Context — image captioning benchmark |
| **Flickr30K** | 30,000 images with 5 captions each |
| **CC3M** | Conceptual Captions 3M — web-crawled image-caption pairs |
| **ShareGPT4V** | GPT-4V generated detailed image descriptions |

> **Note:** Large dataset files (>100 MB) are not included in the repository. See the data processing scripts for download and preparation instructions.

## License

This project is for research purposes.
