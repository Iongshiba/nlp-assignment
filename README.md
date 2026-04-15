# NLP Tokenization and Language Modeling Assignment

This repository compares Word, Character, and BPE tokenization on:

1. One Billion Word
2. WikiText-103
3. Text8
4. Enwik8

It also runs LSTM language-model experiments and generates analysis tables/plots.

## 1. Requirements

- Python 3.9 or newer
- pip
- Optional: conda (if you prefer conda environments)

## 2. Environment Setup

From the repository root:

### Option A: venv (recommended)

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Option B: conda

```bash
conda create -n nlp-assignment python=3.10 -y
conda activate nlp-assignment
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Reproduce Results (End-to-End)

The commands below are the reproducible pipeline used by this project.

### Step 1: Download/prepare raw datasets

```bash
python data/download_data.py --raw-data-dir data/raw
```

Notes:
- One Billion is very large. If needed, skip it during download:

```bash
python data/download_data.py --raw-data-dir data/raw --skip-one-billion
```

### Step 2: Preprocess datasets

```bash
python preprocessing/preprocess.py --raw-data-dir data/raw --output-dir artifacts/processed --dataset all --seed 42
```

Faster preprocessing for One Billion (10% file subset):

```bash
python preprocessing/preprocess.py --raw-data-dir data/raw --output-dir artifacts/processed --dataset one_billion --one-billion-subset 0.1 --seed 42
```

### Step 3: Run tokenizer experiments (Word/Char/BPE)

Example vocab sweep used in this assignment:

```bash
python main.py --processed-dir artifacts/processed --report-dir report --run-tokenizer-experiments --vocab-sizes 2000 8000 16000 32000 36000 50000
```

Outputs are written per dataset under report/<dataset>/{word,char,bpe}.

### Step 4: Run LM experiments

```bash
python main.py --processed-dir artifacts/processed --run-lm-experiments --lm-output-dir artifacts/lm --vocab-sizes 2000 8000 16000 32000 36000 50000
```

Important behavior:
- main.py currently runs LM for text8 and enwik8 only.
- one_billion and wikitext103 are skipped by design in LM orchestration.

### Step 5: Build comparison CSVs and plots

```bash
python analysis/compare_metrics.py --processed-dir artifacts/processed --output-dir artifacts/results --lm-results-dir artifacts/lm --word-vocab-size 50000 --bpe-vocab-sizes 2000 8000 16000 32000 36000 50000
```

## 4. Fast Reproduction (Minimal)

If you only need a quick run for Enwik8 + Text8:

```bash
python data/download_data.py --raw-data-dir data/raw --skip-one-billion --skip-wikitext
python preprocessing/preprocess.py --raw-data-dir data/raw --output-dir artifacts/processed --dataset text8 --seed 42
python preprocessing/preprocess.py --raw-data-dir data/raw --output-dir artifacts/processed --dataset enwik8 --seed 42
python main.py --processed-dir artifacts/processed --report-dir report --run-tokenizer-experiments --run-lm-experiments --lm-output-dir artifacts/lm --vocab-sizes 2000 36000
python analysis/compare_metrics.py --processed-dir artifacts/processed --output-dir artifacts/results --lm-results-dir artifacts/lm --bpe-vocab-sizes 2000 36000
python analysis/plot_enwik8_lm_table.py
```

## 5. Key Output Paths

- Processed splits: artifacts/processed/<dataset>/{train,validation,test}.txt
- Tokenizer metrics JSONs: report/<dataset>/{word,char,bpe}/*_metrics_*.json
- LM outputs: artifacts/lm/<dataset>/<tokenizer_run>/
- Aggregated analysis: artifacts/results/
- Enwik8 4.5 charts script output: report/enwik8/final_analysis/plots/

## 6. Reproducibility Notes

- Seed is fixed to 42 in preprocessing and LM scripts unless overridden.
- Exact training time can vary by CPU/GPU, CUDA, and PyTorch backend settings.
- Downloaded datasets may take significant disk space and time (especially One Billion).
