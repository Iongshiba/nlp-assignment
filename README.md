# Tokenization and NLP Preprocessing Project

This project implements and compares word-level, character-level, and BPE tokenization across four benchmark corpora:

1. One Billion Word Benchmark
2. WikiText-103
3. Text8
4. Enwik8

It also runs a controlled LSTM language modeling experiment using one selected dataset (WikiText-103 or Text8) to compare tokenization effects on perplexity and training speed.

## Folder Layout

```text
project/
├── data/
│   └── download_data.py
├── preprocessing/
│   └── preprocess.py
├── tokenizers_impl/
│   ├── word_tokenizer.py
│   ├── char_tokenizer.py
│   └── bpe_tokenizer.py
├── experiments/
│   ├── lm_model.py
│   └── train_lm.py
├── analysis/
│   ├── compare_metrics.py
│   └── generate_report.py
├── report/
│   └── report_template.md
├── requirements.txt
└── README.md
```

## Setup

Python 3.9+ is required.

```bash
cd project
conda run -n nlp python -m pip install -r requirements.txt
```

## Data Preparation

The repository already includes some raw datasets under `../data`. The script below checks local availability and downloads missing datasets.

```bash
conda run -n nlp python data/download_data.py --raw-data-dir ../data
```

## Preprocessing

Run preprocessing for all datasets:

```bash
conda run -n nlp python preprocessing/preprocess.py --raw-data-dir ../data --output-dir artifacts/processed --dataset all
```

Optional: use a 10% subset of One Billion Word for faster experiments.

```bash
conda run -n nlp python preprocessing/preprocess.py --raw-data-dir ../data --output-dir artifacts/processed --dataset one_billion --one-billion-subset 0.1
```

## Tokenization Metrics and Plots

Compute tokenization metrics for all datasets and generate comparison plots:

```bash
conda run -n nlp python analysis/compare_metrics.py --processed-dir artifacts/processed --output-dir artifacts/results
```

This generates:

- `artifacts/results/tokenization_metrics.csv`
- `artifacts/results/sequence_lengths.csv`
- `artifacts/results/plots/vocab_size_comparison.png`
- `artifacts/results/plots/sequence_length_distribution.png`
- `artifacts/results/plots/compression_ratio_comparison.png`
- `artifacts/results/summary_table.csv`

## Language Model Experiment

Run LSTM LM with one tokenizer at a time:

```bash
conda run -n nlp python experiments/train_lm.py \
  --processed-dir artifacts/processed \
  --dataset wikitext103 \
  --tokenizer word \
  --epochs 5 \
  --batch-size 32 \
  --sequence-length 128 \
  --output-dir artifacts/lm
```

BPE example:

```bash
conda run -n nlp python experiments/train_lm.py \
  --processed-dir artifacts/processed \
  --dataset text8 \
  --tokenizer bpe \
  --bpe-vocab-size 10000 \
  --epochs 5 \
  --batch-size 32 \
  --sequence-length 128 \
  --output-dir artifacts/lm
```

After running all tokenizer variants, build LM comparison plot:

```bash
conda run -n nlp python analysis/compare_metrics.py \
  --processed-dir artifacts/processed \
  --output-dir artifacts/results \
  --lm-results-dir artifacts/lm
```

## Report Generation

Generate report markdown and optional PDF (requires Pandoc installed on system):

```bash
conda run -n nlp python analysis/generate_report.py \
  --results-dir artifacts/results \
  --lm-results-dir artifacts/lm \
  --output-markdown report/final_report.md \
  --output-pdf report/final_report.pdf
```

If Pandoc is unavailable, markdown output is still generated.

## Reproducibility

- Global random seed: 42
- All preprocessing and tokenization steps are deterministic where possible.
