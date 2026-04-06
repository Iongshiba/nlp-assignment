from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List

SEED = 42


def normalize_text(text: str) -> str:
    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "…": "...",
    }

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    text = text.lower()
    text = text.replace("\t", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Keep lowercase letters, digits, and a compact punctuation set.
    text = re.sub(r"[^a-z0-9\s\.,!\?;:'\"\-\(\)]", " ", text)

    # Normalize repeated whitespace and strip.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _read_lines_from_file(path: Path, max_lines: int | None = None) -> List[str]:
    lines: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            line = line.strip()
            if line:
                lines.append(line)
                if max_lines is not None and len(lines) >= max_lines:
                    break
    return lines


def _read_lines_from_files(
    files: List[Path], max_lines: int | None = None
) -> List[str]:
    lines: List[str] = []
    for file_path in files:
        remaining = None if max_lines is None else max_lines - len(lines)
        if remaining is not None and remaining <= 0:
            break
        lines.extend(_read_lines_from_file(file_path, max_lines=remaining))
    return lines


def _maybe_limit(samples: List[str], max_samples: int | None) -> List[str]:
    if max_samples is None:
        return samples
    return samples[:max_samples]


def load_one_billion(
    raw_data_dir: Path,
    subset_fraction: float,
    seed: int,
    max_samples_per_split: int | None = None,
) -> Dict[str, List[str]]:
    dataset_dir = raw_data_dir / "1-billion-word-language-modeling-benchmark-r13output"
    train_dir = dataset_dir / "training-monolingual.tokenized.shuffled"
    heldout_dir = dataset_dir / "heldout-monolingual.tokenized.shuffled"

    if not train_dir.exists() or not heldout_dir.exists():
        raise FileNotFoundError("One Billion Word dataset directories were not found.")

    train_files = sorted([p for p in train_dir.glob("news.en-*-of-*") if p.is_file()])
    heldout_files = sorted(
        [p for p in heldout_dir.glob("news.en.heldout-*-of-*") if p.is_file()]
    )

    if not train_files:
        raise FileNotFoundError("No One Billion training files found.")
    if not heldout_files:
        raise FileNotFoundError("No One Billion heldout files found.")

    rng = random.Random(seed)
    if subset_fraction < 1.0:
        selected_count = max(1, math.ceil(len(train_files) * subset_fraction))
        train_files = sorted(rng.sample(train_files, selected_count))

    train_samples = _read_lines_from_files(train_files, max_lines=max_samples_per_split)

    midpoint = len(heldout_files) // 2
    valid_files = heldout_files[:midpoint]
    test_files = heldout_files[midpoint:]

    valid_samples = _read_lines_from_files(valid_files, max_lines=max_samples_per_split)

    test_samples = _read_lines_from_files(test_files, max_lines=max_samples_per_split)

    return {
        "train": train_samples,
        "validation": valid_samples,
        "test": test_samples,
    }


def load_wikitext103(
    raw_data_dir: Path, max_samples_per_split: int | None = None
) -> Dict[str, List[str]]:
    dataset_dir = raw_data_dir / "wikitext-103"
    split_map = {
        "train": dataset_dir / "wiki.train.tokens",
        "validation": dataset_dir / "wiki.valid.tokens",
        "test": dataset_dir / "wiki.test.tokens",
    }

    output: Dict[str, List[str]] = {}
    for split_name, file_path in split_map.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing WikiText file: {file_path}")
        output[split_name] = _read_lines_from_file(
            file_path, max_lines=max_samples_per_split
        )

    return output


def _split_samples(
    samples: List[str], train_ratio: float, valid_ratio: float
) -> Dict[str, List[str]]:
    n_total = len(samples)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)

    train = samples[:n_train]
    validation = samples[n_train : n_train + n_valid]
    test = samples[n_train + n_valid :]

    return {
        "train": train,
        "validation": validation,
        "test": test,
    }


def load_text8(
    raw_data_dir: Path,
    seed: int,
    chunk_words: int,
    max_samples_per_split: int | None = None,
) -> Dict[str, List[str]]:
    text8_file = raw_data_dir / "text8" / "text8"
    if not text8_file.exists():
        raise FileNotFoundError("Text8 file was not found.")

    content = text8_file.read_text(encoding="utf-8", errors="ignore")
    words = content.split()

    samples = [
        " ".join(words[i : i + chunk_words]) for i in range(0, len(words), chunk_words)
    ]
    samples = [sample for sample in samples if sample.strip()]

    rng = random.Random(seed)
    rng.shuffle(samples)
    split_samples = _split_samples(samples, train_ratio=0.9, valid_ratio=0.05)
    if max_samples_per_split is not None:
        for split_name in split_samples:
            split_samples[split_name] = split_samples[split_name][
                :max_samples_per_split
            ]
    return split_samples


def load_enwik8(
    raw_data_dir: Path,
    seed: int,
    chunk_chars: int,
    max_samples_per_split: int | None = None,
) -> Dict[str, List[str]]:
    candidate_paths = [
        raw_data_dir / "enwik8" / "enwik8",
        raw_data_dir / "enwik8",
    ]

    file_path = None
    for candidate in candidate_paths:
        if candidate.exists() and candidate.is_file():
            file_path = candidate
            break

    if file_path is None:
        raise FileNotFoundError("Enwik8 file was not found.")

    data = file_path.read_bytes().decode("latin-1", errors="ignore")
    samples = [data[i : i + chunk_chars] for i in range(0, len(data), chunk_chars)]
    samples = [sample for sample in samples if sample.strip()]

    rng = random.Random(seed)
    rng.shuffle(samples)
    split_samples = _split_samples(samples, train_ratio=0.9, valid_ratio=0.05)
    if max_samples_per_split is not None:
        for split_name in split_samples:
            split_samples[split_name] = split_samples[split_name][
                :max_samples_per_split
            ]
    return split_samples


def preprocess_dataset(
    splits: Dict[str, List[str]], max_samples_per_split: int | None = None
) -> Dict[str, List[str]]:
    processed: Dict[str, List[str]] = {}
    for split_name, samples in splits.items():
        normalized = [normalize_text(sample) for sample in samples]
        normalized = [sample for sample in normalized if sample]
        normalized = _maybe_limit(normalized, max_samples_per_split)
        processed[split_name] = normalized
    return processed


def save_processed_dataset(
    output_dir: Path,
    dataset_name: str,
    splits: Dict[str, List[str]],
    metadata: Dict[str, object],
) -> None:
    dataset_out_dir = output_dir / dataset_name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, samples in splits.items():
        out_file = dataset_out_dir / f"{split_name}.txt"
        with out_file.open("w", encoding="utf-8") as fout:
            for sample in samples:
                fout.write(sample + "\n")

    metadata_file = dataset_out_dir / "metadata.json"
    with metadata_file.open("w", encoding="utf-8") as fout:
        json.dump(metadata, fout, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess datasets for tokenization experiments."
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("../data"),
        help="Directory containing raw datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/processed"),
        help="Directory to store processed outputs.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "one_billion", "wikitext103", "text8", "enwik8"],
        help="Dataset to preprocess.",
    )
    parser.add_argument(
        "--one-billion-subset",
        type=float,
        default=1.0,
        help="Fraction of One Billion training files to use (0, 1].",
    )
    parser.add_argument(
        "--text8-chunk-words",
        type=int,
        default=200,
        help="Words per sample chunk for Text8.",
    )
    parser.add_argument(
        "--enwik8-chunk-chars",
        type=int,
        default=1000,
        help="Characters per sample chunk for Enwik8.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Optional cap for each split size after preprocessing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.one_billion_subset <= 1.0:
        raise ValueError("--one-billion-subset must be in (0, 1].")

    raw_data_dir = args.raw_data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = {
        "one_billion": lambda: load_one_billion(
            raw_data_dir,
            args.one_billion_subset,
            args.seed,
            max_samples_per_split=args.max_samples_per_split,
        ),
        "wikitext103": lambda: load_wikitext103(
            raw_data_dir, max_samples_per_split=args.max_samples_per_split
        ),
        "text8": lambda: load_text8(
            raw_data_dir,
            args.seed,
            args.text8_chunk_words,
            max_samples_per_split=args.max_samples_per_split,
        ),
        "enwik8": lambda: load_enwik8(
            raw_data_dir,
            args.seed,
            args.enwik8_chunk_chars,
            max_samples_per_split=args.max_samples_per_split,
        ),
    }

    selected = list(loaders.keys()) if args.dataset == "all" else [args.dataset]

    for dataset_name in selected:
        print(f"Preprocessing {dataset_name}...")
        raw_splits = loaders[dataset_name]()
        processed_splits = preprocess_dataset(
            raw_splits, max_samples_per_split=args.max_samples_per_split
        )

        metadata = {
            "dataset": dataset_name,
            "seed": args.seed,
            "normalization": {
                "lowercase": True,
                "punctuation": "kept compact ASCII punctuation . , ! ? ; : ' \" - ( )",
                "non_ascii": "removed or replaced",
                "whitespace": "collapsed to single spaces",
            },
            "num_samples": {k: len(v) for k, v in processed_splits.items()},
            "one_billion_subset_fraction": (
                args.one_billion_subset if dataset_name == "one_billion" else None
            ),
        }

        save_processed_dataset(output_dir, dataset_name, processed_splits, metadata)
        print(f"Saved processed data for {dataset_name} to {output_dir / dataset_name}")


if __name__ == "__main__":
    main()
