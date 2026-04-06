from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def _to_markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "No data available."
    return df.head(max_rows).to_markdown(index=False)


def _relative_path(target: Path, base: Path) -> str:
    if not target.exists():
        return str(target)
    return os.path.relpath(target.resolve(), start=base.resolve().parent)


def build_report(
    results_dir: Path,
    lm_results_dir: Path | None,
    output_markdown: Path,
) -> str:
    metrics_path = results_dir / "tokenization_metrics.csv"
    summary_path = results_dir / "summary_table.csv"
    sequence_path = results_dir / "sequence_lengths.csv"

    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    summary_df = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    sequence_df = (
        pd.read_csv(sequence_path) if sequence_path.exists() else pd.DataFrame()
    )

    datasets = (
        sorted(metrics_df["dataset"].unique().tolist()) if not metrics_df.empty else []
    )
    tokenizers = (
        sorted(metrics_df["tokenizer"].unique().tolist())
        if not metrics_df.empty
        else []
    )

    lm_overview = pd.DataFrame()
    if lm_results_dir is not None and lm_results_dir.exists():
        records = []
        for metrics_file in lm_results_dir.glob("*/*/metrics.csv"):
            df = pd.read_csv(metrics_file)
            if df.empty:
                continue
            records.append(
                {
                    "dataset": metrics_file.parents[1].name,
                    "tokenizer": metrics_file.parents[0].name,
                    "final_validation_perplexity": float(
                        df["validation_perplexity"].iloc[-1]
                    ),
                    "avg_epoch_time_seconds": float(df["epoch_time_seconds"].mean()),
                }
            )
        lm_overview = pd.DataFrame.from_records(records)

    plot_dir = results_dir / "plots"
    vocab_plot = plot_dir / "vocab_size_comparison.png"
    seq_plot = plot_dir / "sequence_length_distribution.png"
    ppl_plot = plot_dir / "perplexity_vs_epoch.png"

    lines = []
    lines.append("# Tokenization and Preprocessing Report")
    lines.append("")
    lines.append("## 1. Introduction")
    lines.append(
        "Tokenization strongly affects language modeling quality, memory usage, and runtime. "
        "This report compares word-level, character-level, and Byte Pair Encoding (BPE) tokenization "
        "under a shared preprocessing pipeline and a fixed LSTM architecture."
    )
    lines.append("")

    lines.append("## 2. Background")
    lines.append(
        "Word-level tokenization maps text into lexical units and introduces an <UNK> token for out-of-vocabulary words."
    )
    lines.append(
        "Character-level tokenization avoids OOV at the word level but yields longer sequences."
    )
    lines.append("BPE learns subword units by repeatedly merging frequent token pairs.")
    lines.append("")
    lines.append("Useful formulas:")
    lines.append("")
    lines.append(
        "- Compression ratio: $\\text{CR} = \\frac{\\text{#tokens}}{\\text{#characters}}$"
    )
    lines.append(
        "- OOV rate: $\\text{OOV} = \\frac{\\text{#unknown tokens}}{\\text{#test tokens}} \\times 100$"
    )
    lines.append(
        "- Perplexity: $\\text{PPL} = e^{\\mathcal{L}}$, where $\\mathcal{L}$ is average cross-entropy loss"
    )
    lines.append("")

    lines.append("## 3. Datasets")
    lines.append(
        "The study covers four datasets: One Billion Word Benchmark, WikiText-103, Text8, and Enwik8."
    )
    lines.append(
        "If compute is limited, the One Billion Word training set may use a 10% random subset."
    )
    lines.append("")
    if datasets:
        lines.append("Detected processed datasets:")
        lines.append("")
        for name in datasets:
            lines.append(f"- {name}")
        lines.append("")

    lines.append("## 4. Methodology")
    lines.append("### 4.1 Preprocessing")
    lines.append("- Lowercasing")
    lines.append("- Normalized punctuation to compact ASCII symbols")
    lines.append("- Removed unsupported special characters")
    lines.append("- Collapsed repeated whitespace")
    lines.append(
        "- Used provided train/validation/test splits when available; otherwise generated 90/5/5 splits"
    )
    lines.append("")

    lines.append("### 4.2 Tokenizer Setup")
    lines.append("- Word tokenizer: max vocabulary 50,000 with <UNK>")
    lines.append("- Character tokenizer: character vocabulary from training split")
    lines.append("- BPE tokenizer: vocab sizes 1,000 / 5,000 / 10,000 / 30,000")
    lines.append("")

    lines.append("### 4.3 Language Model")
    lines.append("- Architecture: 2-layer LSTM, embedding size 128, hidden size 256")
    lines.append("- Optimizer: Adam (lr = 0.001)")
    lines.append("- Batch size: 32, sequence length: 128")
    lines.append("- Epochs: 5")
    lines.append("")

    lines.append("## 5. Results")
    lines.append("### 5.1 Summary Comparison Table")
    lines.append("")
    lines.append(_to_markdown_table(summary_df, max_rows=60))
    lines.append("")

    lines.append("### 5.2 Vocabulary Size Comparison")
    lines.append("")
    if vocab_plot.exists():
        lines.append(
            f"![Vocabulary Size Comparison]({_relative_path(vocab_plot, output_markdown)})"
        )
    else:
        lines.append(
            "Vocabulary plot not found. Run analysis/compare_metrics.py first."
        )
    lines.append("")

    lines.append("### 5.3 Sequence Length Distribution")
    lines.append("")
    if seq_plot.exists():
        lines.append(
            f"![Sequence Length Distribution]({_relative_path(seq_plot, output_markdown)})"
        )
    else:
        lines.append(
            "Sequence length plot not found. Run analysis/compare_metrics.py first."
        )
    lines.append("")

    lines.append("### 5.4 Perplexity Curves")
    lines.append("")
    if ppl_plot.exists():
        lines.append(
            f"![Perplexity vs Epoch]({_relative_path(ppl_plot, output_markdown)})"
        )
    else:
        lines.append(
            "Perplexity curve not found. Run LM training and re-run analysis/compare_metrics.py with --lm-results-dir."
        )
    lines.append("")

    if not lm_overview.empty:
        lines.append("LM final metrics:")
        lines.append("")
        lines.append(_to_markdown_table(lm_overview, max_rows=30))
        lines.append("")

    lines.append("## 6. Discussion")
    lines.append(
        "Word tokenization usually has shorter sequences than characters but can suffer from OOV. "
        "Character tokenization has no word-level OOV but increases sequence length and training cost. "
        "BPE balances both by reducing OOV pressure while maintaining moderate sequence lengths."
    )
    lines.append("")

    lines.append("## 7. Limitations and Conclusion")
    lines.append(
        "Results depend on preprocessing choices, subset size, and compute budget. "
        "Future work can include Transformer-based models, multilingual corpora, and tokenizer regularization experiments."
    )
    lines.append("")

    lines.append("## Appendix")
    lines.append("### Raw Tokenization Metrics")
    lines.append("")
    lines.append(_to_markdown_table(metrics_df, max_rows=100))
    lines.append("")

    if not sequence_df.empty:
        lines.append("### Sequence Length Sample")
        lines.append("")
        lines.append(_to_markdown_table(sequence_df.head(50), max_rows=50))

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate markdown and PDF report from experiment outputs."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--lm-results-dir", type=Path, default=None)
    parser.add_argument(
        "--output-markdown", type=Path, default=Path("report/final_report.md")
    )
    parser.add_argument(
        "--output-pdf", type=Path, default=Path("report/final_report.pdf")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    markdown_text = build_report(
        results_dir=args.results_dir.resolve(),
        lm_results_dir=args.lm_results_dir.resolve() if args.lm_results_dir else None,
        output_markdown=args.output_markdown.resolve(),
    )

    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown_text, encoding="utf-8")
    print(f"Saved markdown report to {args.output_markdown}")

    try:
        import pypandoc

        args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
        pypandoc.convert_file(
            str(args.output_markdown),
            "pdf",
            outputfile=str(args.output_pdf),
            extra_args=["--standalone"],
        )
        print(f"Saved PDF report to {args.output_pdf}")
    except Exception as exc:  # noqa: BLE001
        print("PDF export skipped. Install Pandoc and ensure pypandoc can find it.")
        print(f"Reason: {exc}")


if __name__ == "__main__":
    main()
