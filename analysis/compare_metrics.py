from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from tokenizers_impl.bpe_tokenizer import BPETokenizerConfig, BPETokenizerWrapper
from tokenizers_impl.char_tokenizer import CharTokenizer, CharTokenizerConfig
from tokenizers_impl.word_tokenizer import WordTokenizer, WordTokenizerConfig


def read_split(
    dataset_dir: Path, split_name: str, max_samples: int | None = None
) -> List[str]:
    split_path = dataset_dir / f"{split_name}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    samples: List[str] = []
    with split_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                samples.append(line)
                if max_samples is not None and len(samples) >= max_samples:
                    break
    return samples


def compute_word_stats(
    tokenizer: WordTokenizer, samples: List[str]
) -> Tuple[Dict[str, float], List[int], int]:
    sequence_lengths: List[int] = []
    total_tokens = 0
    total_chars = 0
    oov_tokens = 0

    for sample in samples:
        tokens = tokenizer.tokenize(sample)
        seq_len = len(tokens)
        sequence_lengths.append(seq_len)
        total_tokens += seq_len
        total_chars += len(sample)

        for token in tokens:
            if token not in tokenizer.token_to_id:
                oov_tokens += 1

    avg_sequence_length = (
        (sum(sequence_lengths) / len(sequence_lengths)) if sequence_lengths else 0.0
    )
    oov_rate = (oov_tokens / total_tokens * 100.0) if total_tokens > 0 else 0.0
    compression_ratio = (total_tokens / total_chars) if total_chars > 0 else 0.0

    metrics = {
        "vocab_size": float(tokenizer.vocab_size()),
        "avg_sequence_length": avg_sequence_length,
        "oov_rate": oov_rate,
        "compression_ratio": compression_ratio,
    }
    return metrics, sequence_lengths, total_tokens


def compute_char_stats(
    tokenizer: CharTokenizer, samples: List[str]
) -> Tuple[Dict[str, float], List[int], int]:
    sequence_lengths = [len(sample) for sample in samples]
    total_tokens = sum(sequence_lengths)
    total_chars = sum(len(sample) for sample in samples)

    avg_sequence_length = (
        (sum(sequence_lengths) / len(sequence_lengths)) if sequence_lengths else 0.0
    )
    compression_ratio = (total_tokens / total_chars) if total_chars > 0 else 0.0

    metrics = {
        "vocab_size": float(tokenizer.vocab_size()),
        "avg_sequence_length": avg_sequence_length,
        "oov_rate": np.nan,
        "compression_ratio": compression_ratio,
    }
    return metrics, sequence_lengths, total_tokens


def compute_bpe_stats(
    tokenizer: BPETokenizerWrapper, samples: List[str]
) -> Tuple[Dict[str, float], List[int], int]:
    sequence_lengths: List[int] = []
    total_tokens = 0
    total_chars = 0

    for sample in samples:
        token_ids = tokenizer.encode(sample)
        seq_len = len(token_ids)
        sequence_lengths.append(seq_len)
        total_tokens += seq_len
        total_chars += len(sample)

    avg_sequence_length = (
        (sum(sequence_lengths) / len(sequence_lengths)) if sequence_lengths else 0.0
    )
    compression_ratio = (total_tokens / total_chars) if total_chars > 0 else 0.0

    metrics = {
        "vocab_size": float(tokenizer.vocab_size()),
        "avg_sequence_length": avg_sequence_length,
        "oov_rate": np.nan,
        "compression_ratio": compression_ratio,
    }
    return metrics, sequence_lengths, total_tokens


def plot_vocab_size(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    pivot_df = metrics_df.pivot_table(
        index="dataset", columns="tokenizer", values="vocab_size", aggfunc="mean"
    )
    pivot_df.plot(kind="bar", ax=ax)
    ax.set_title("Vocabulary Size Comparison Across Datasets and Tokenizers")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Vocabulary Size")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Tokenizer", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_sequence_distribution(sequence_df: pd.DataFrame, output_path: Path) -> None:
    datasets = sorted(sequence_df["dataset"].unique())
    n_rows = len(datasets)

    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows), sharex=False)
    if n_rows == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        subset = sequence_df[sequence_df["dataset"] == dataset]
        for tokenizer_name in sorted(subset["tokenizer"].unique()):
            lengths = subset[subset["tokenizer"] == tokenizer_name][
                "sequence_length"
            ].values
            if len(lengths) == 0:
                continue
            ax.hist(lengths, bins=60, alpha=0.35, label=tokenizer_name, density=True)

        ax.set_title(f"Sequence Length Distribution - {dataset}")
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Density")
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_compression(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    pivot_df = metrics_df.pivot_table(
        index="dataset", columns="tokenizer", values="compression_ratio", aggfunc="mean"
    )
    pivot_df.plot(kind="bar", ax=ax)
    ax.set_title("Compression Ratio (Tokens per Character)")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Tokens / Character")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Tokenizer", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def load_lm_metrics(lm_results_dir: Path) -> pd.DataFrame:
    records = []
    if not lm_results_dir.exists():
        return pd.DataFrame()

    for metrics_file in lm_results_dir.glob("*/*/metrics.csv"):
        df = pd.read_csv(metrics_file)
        if df.empty:
            continue

        dataset = metrics_file.parents[1].name
        tokenizer = metrics_file.parents[0].name

        for _, row in df.iterrows():
            records.append(
                {
                    "dataset": dataset,
                    "tokenizer": tokenizer,
                    "epoch": int(row["epoch"]),
                    "validation_perplexity": float(row["validation_perplexity"]),
                    "epoch_time_seconds": float(row["epoch_time_seconds"]),
                }
            )

    return pd.DataFrame.from_records(records)


def plot_perplexity_curve(lm_df: pd.DataFrame, output_path: Path) -> None:
    if lm_df.empty:
        return

    datasets = sorted(lm_df["dataset"].unique())
    n_rows = len(datasets)

    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        subset = lm_df[lm_df["dataset"] == dataset]
        for tokenizer_name in sorted(subset["tokenizer"].unique()):
            run = subset[subset["tokenizer"] == tokenizer_name].sort_values("epoch")
            ax.plot(
                run["epoch"],
                run["validation_perplexity"],
                marker="o",
                label=tokenizer_name,
            )

        ax.set_title(f"Validation Perplexity vs Epoch - {dataset}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Perplexity")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare tokenization metrics and generate plots."
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("artifacts/processed")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names. Defaults to all directories in processed-dir.",
    )
    parser.add_argument("--word-vocab-size", type=int, default=50000)
    parser.add_argument(
        "--bpe-vocab-sizes", type=int, nargs="+", default=[1000, 5000, 10000, 30000]
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-sequence-samples", type=int, default=5000)
    parser.add_argument("--lm-results-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = args.processed_dir.resolve()
    output_dir = args.output_dir.resolve()
    plot_dir = output_dir / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets is None:
        dataset_names = sorted([p.name for p in processed_dir.iterdir() if p.is_dir()])
    else:
        dataset_names = args.datasets

    metric_rows: List[Dict[str, object]] = []
    sequence_rows: List[Dict[str, object]] = []

    for dataset_name in dataset_names:
        print(f"Processing dataset: {dataset_name}")
        dataset_dir = processed_dir / dataset_name
        train_samples = read_split(
            dataset_dir, "train", max_samples=args.max_train_samples
        )
        test_samples = read_split(
            dataset_dir, "test", max_samples=args.max_eval_samples
        )

        if not train_samples or not test_samples:
            print(f"Skipping {dataset_name} because train/test splits are empty.")
            continue

        word_tokenizer = WordTokenizer(
            WordTokenizerConfig(vocab_size=args.word_vocab_size)
        )
        word_tokenizer.train(train_samples)
        word_metrics, word_seq_lengths, word_total_tokens = compute_word_stats(
            word_tokenizer, test_samples
        )

        metric_rows.append(
            {
                "dataset": dataset_name,
                "tokenizer": "word",
                "tokenizer_family": "word",
                "vocab_size": word_metrics["vocab_size"],
                "avg_sequence_length": word_metrics["avg_sequence_length"],
                "oov_rate": word_metrics["oov_rate"],
                "compression_ratio": word_metrics["compression_ratio"],
                "compression_vs_word": 1.0,
            }
        )

        for seq_len in word_seq_lengths[: args.max_sequence_samples]:
            sequence_rows.append(
                {
                    "dataset": dataset_name,
                    "tokenizer": "word",
                    "sequence_length": seq_len,
                }
            )

        char_tokenizer = CharTokenizer(CharTokenizerConfig(max_vocab_size=None))
        char_tokenizer.train(train_samples)
        char_metrics, char_seq_lengths, _ = compute_char_stats(
            char_tokenizer, test_samples
        )

        metric_rows.append(
            {
                "dataset": dataset_name,
                "tokenizer": "char",
                "tokenizer_family": "char",
                "vocab_size": char_metrics["vocab_size"],
                "avg_sequence_length": char_metrics["avg_sequence_length"],
                "oov_rate": np.nan,
                "compression_ratio": char_metrics["compression_ratio"],
                "compression_vs_word": (
                    (
                        char_metrics["compression_ratio"]
                        / word_metrics["compression_ratio"]
                    )
                    if word_metrics["compression_ratio"] > 0
                    else np.nan
                ),
            }
        )

        for seq_len in char_seq_lengths[: args.max_sequence_samples]:
            sequence_rows.append(
                {
                    "dataset": dataset_name,
                    "tokenizer": "char",
                    "sequence_length": seq_len,
                }
            )

        for bpe_vocab_size in tqdm(args.bpe_vocab_sizes, desc=f"BPE on {dataset_name}"):
            bpe_tokenizer = BPETokenizerWrapper(
                BPETokenizerConfig(vocab_size=bpe_vocab_size)
            )
            bpe_tokenizer.train(train_samples)
            bpe_metrics, bpe_seq_lengths, bpe_total_tokens = compute_bpe_stats(
                bpe_tokenizer, test_samples
            )

            tokenizer_name = f"bpe_{bpe_vocab_size}"
            metric_rows.append(
                {
                    "dataset": dataset_name,
                    "tokenizer": tokenizer_name,
                    "tokenizer_family": "bpe",
                    "vocab_size": bpe_metrics["vocab_size"],
                    "avg_sequence_length": bpe_metrics["avg_sequence_length"],
                    "oov_rate": np.nan,
                    "compression_ratio": bpe_metrics["compression_ratio"],
                    "compression_vs_word": (
                        (bpe_total_tokens / word_total_tokens)
                        if word_total_tokens > 0
                        else np.nan
                    ),
                }
            )

            for seq_len in bpe_seq_lengths[: args.max_sequence_samples]:
                sequence_rows.append(
                    {
                        "dataset": dataset_name,
                        "tokenizer": tokenizer_name,
                        "sequence_length": seq_len,
                    }
                )

    metrics_df = pd.DataFrame.from_records(metric_rows)
    sequence_df = pd.DataFrame.from_records(sequence_rows)

    metrics_path = output_dir / "tokenization_metrics.csv"
    sequence_path = output_dir / "sequence_lengths.csv"
    metrics_df.to_csv(metrics_path, index=False)
    sequence_df.to_csv(sequence_path, index=False)

    if not metrics_df.empty:
        plot_vocab_size(metrics_df, plot_dir / "vocab_size_comparison.png")
        plot_sequence_distribution(
            sequence_df, plot_dir / "sequence_length_distribution.png"
        )
        plot_compression(metrics_df, plot_dir / "compression_ratio_comparison.png")

    summary_df = metrics_df.copy()

    if args.lm_results_dir is not None:
        lm_df = load_lm_metrics(args.lm_results_dir.resolve())
        if not lm_df.empty:
            plot_perplexity_curve(lm_df, plot_dir / "perplexity_vs_epoch.png")

            final_lm = (
                lm_df.sort_values("epoch")
                .groupby(["dataset", "tokenizer"], as_index=False)
                .agg(
                    final_validation_perplexity=("validation_perplexity", "last"),
                    avg_epoch_time_seconds=("epoch_time_seconds", "mean"),
                )
            )

            summary_df = summary_df.merge(
                final_lm, how="left", on=["dataset", "tokenizer"]
            )

    summary_path = output_dir / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved tokenization metrics to {metrics_path}")
    print(f"Saved sequence length stats to {sequence_path}")
    print(f"Saved summary table to {summary_path}")
    print(f"Plots are in {plot_dir}")


if __name__ == "__main__":
    main()
