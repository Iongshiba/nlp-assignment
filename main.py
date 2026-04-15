from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tokenizer and LM experiments for all datasets."
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("artifacts/processed")
    )
    parser.add_argument("--report-dir", type=Path, default=Path("report"))
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=[36000],
    )
    parser.add_argument(
        "--run-tokenizer-experiments",
        action="store_true",
        help="Run tokenizer experiments for word/char/bpe.",
    )
    parser.add_argument(
        "--run-lm-experiments",
        action="store_true",
        help="Run LM training experiments for all datasets and tokenizer variants.",
    )
    parser.add_argument("--lm-output-dir", type=Path, default=Path("artifacts/lm"))
    return parser.parse_args()


def _run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _iter_dataset_dirs(processed_dir: Path) -> Iterable[Path]:
    for dataset_dir in sorted(processed_dir.iterdir()):
        if dataset_dir.is_dir():
            yield dataset_dir


def _run_tokenizer_experiments(
    root_dir: Path, processed_dir: Path, report_dir: Path, vocab_sizes: list[str]
) -> None:
    for dataset_dir in _iter_dataset_dirs(processed_dir):
        dataset_name = dataset_dir.name
        print("=" * 50)
        print(f"Running tokenizer experiments for dataset: {dataset_name}")

        word_out = report_dir / dataset_name / "word"
        char_out = report_dir / dataset_name / "char"
        bpe_out = report_dir / dataset_name / "bpe"

        word_out.mkdir(parents=True, exist_ok=True)
        char_out.mkdir(parents=True, exist_ok=True)
        bpe_out.mkdir(parents=True, exist_ok=True)

        _run_command(
            [
                sys.executable,
                str(root_dir / "tokenizers_impl" / "word_tokenizer.py"),
                "--processed-dataset-dir",
                str(dataset_dir),
                "--output-dir",
                str(word_out),
                "--vocab-sizes",
                *vocab_sizes,
            ]
        )

        _run_command(
            [
                sys.executable,
                str(root_dir / "tokenizers_impl" / "char_tokenizer.py"),
                "--processed-dataset-dir",
                str(dataset_dir),
                "--output-dir",
                str(char_out),
                "--vocab-sizes",
                *vocab_sizes,
            ]
        )

        _run_command(
            [
                sys.executable,
                str(root_dir / "tokenizers_impl" / "bpe_tokenizer.py"),
                "--processed-dataset-dir",
                str(dataset_dir),
                "--output-dir",
                str(bpe_out),
                "--vocab-sizes",
                *vocab_sizes,
            ]
        )

    print("All tokenizer experiments completed.")


def _run_lm_experiments(
    root_dir: Path, processed_dir: Path, lm_output_dir: Path, vocab_sizes: list[str]
) -> None:
    lm_script = root_dir / "experiments" / "train_lm.py"
    for dataset_dir in _iter_dataset_dirs(processed_dir):
        dataset_name = dataset_dir.name
        if dataset_name in ["one_billion", "wikitext103"]:
            continue
        print("=" * 50)
        print(f"Running LM experiments for dataset: {dataset_name}")

        # for vocab_size in vocab_sizes:
        #     _run_command(
        #         [
        #             sys.executable,
        #             str(lm_script),
        #             "--processed-dir",
        #             str(processed_dir),
        #             "--dataset",
        #             dataset_name,
        #             "--tokenizer",
        #             "word",
        #             "--word-vocab-size",
        #             vocab_size,
        #             "--epochs",
        #             "3",
        #             "--embedding-size",
        #             "128",
        #             "--hidden-size",
        #             "256",
        #             "--num-layers",
        #             "2",
        #             "--learning-rate",
        #             "0.001",
        #             "--batch-size",
        #             "32",
        #             "--sequence-length",
        #             "128",
        #             "--output-dir",
        #             str(lm_output_dir),
        #         ]
        #     )

        for vocab_size in vocab_sizes:
            _run_command(
                [
                    sys.executable,
                    str(lm_script),
                    "--processed-dir",
                    str(processed_dir),
                    "--dataset",
                    dataset_name,
                    "--tokenizer",
                    "char",
                    "--char-vocab-size",
                    vocab_size,
                    "--epochs",
                    "3",
                    "--embedding-size",
                    "128",
                    "--hidden-size",
                    "256",
                    "--num-layers",
                    "2",
                    "--learning-rate",
                    "0.001",
                    "--batch-size",
                    "32",
                    "--sequence-length",
                    "128",
                    "--output-dir",
                    str(lm_output_dir),
                ]
            )

        # for vocab_size in vocab_sizes:
        #     _run_command(
        #         [
        #             sys.executable,
        #             str(lm_script),
        #             "--processed-dir",
        #             str(processed_dir),
        #             "--dataset",
        #             dataset_name,
        #             "--tokenizer",
        #             "bpe",
        #             "--bpe-vocab-size",
        #             vocab_size,
        #             "--epochs",
        #             "3",
        #             "--embedding-size",
        #             "128",
        #             "--hidden-size",
        #             "256",
        #             "--num-layers",
        #             "2",
        #             "--learning-rate",
        #             "0.001",
        #             "--batch-size",
        #             "32",
        #             "--sequence-length",
        #             "128",
        #             "--output-dir",
        #             str(lm_output_dir),
        #         ]
        #     )

    print("All LM experiments completed.")


def main() -> None:
    args = parse_args()

    root_dir = Path(__file__).resolve().parent
    processed_dir = (root_dir / args.processed_dir).resolve()
    report_dir = (root_dir / args.report_dir).resolve()

    if not processed_dir.exists() or not processed_dir.is_dir():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    vocab_sizes = [str(vocab_size) for vocab_size in args.vocab_sizes]
    run_tokenizer = args.run_tokenizer_experiments
    run_lm = args.run_lm_experiments
    if not run_tokenizer and not run_lm:
        run_tokenizer = True

    if run_tokenizer:
        _run_tokenizer_experiments(root_dir, processed_dir, report_dir, vocab_sizes)

    if run_lm:
        lm_output_dir = (root_dir / args.lm_output_dir).resolve()
        _run_lm_experiments(root_dir, processed_dir, lm_output_dir, vocab_sizes)


if __name__ == "__main__":
    main()
