from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen

from tqdm import tqdm

ONE_BILLION_URL = "http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz"
TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"


def _download_with_progress(
    url: str, target_path: Path, chunk_size: int = 1024 * 128
) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        desc = f"Downloading {target_path.name}"
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
            with target_path.open("wb") as fout:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    fout.write(chunk)
                    pbar.update(len(chunk))


def _extract_tar_gz(archive_path: Path, output_dir: Path) -> None:
    print(f"Extracting {archive_path} -> {output_dir}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=output_dir)


def _extract_zip(archive_path: Path, output_dir: Path) -> None:
    print(f"Extracting {archive_path} -> {output_dir}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(path=output_dir)


def _has_files(paths: Iterable[Path]) -> bool:
    return all(path.exists() for path in paths)


def ensure_one_billion(raw_data_dir: Path) -> None:
    one_billion_dir = (
        raw_data_dir / "1-billion-word-language-modeling-benchmark-r13output"
    )
    train_dir = one_billion_dir / "training-monolingual.tokenized.shuffled"
    heldout_dir = one_billion_dir / "heldout-monolingual.tokenized.shuffled"

    if _has_files([train_dir, heldout_dir]):
        print("One Billion Word Benchmark already exists locally. Skipping download.")
        return

    archive_path = (
        raw_data_dir / "1-billion-word-language-modeling-benchmark-r13output.tar.gz"
    )
    if not archive_path.exists():
        print("One Billion archive not found locally. Downloading...")
        _download_with_progress(ONE_BILLION_URL, archive_path)

    _extract_tar_gz(archive_path, raw_data_dir)


def ensure_wikitext(raw_data_dir: Path) -> None:
    wikitext_dir = raw_data_dir / "wikitext-103"
    expected_files = [
        wikitext_dir / "wiki.train.tokens",
        wikitext_dir / "wiki.valid.tokens",
        wikitext_dir / "wiki.test.tokens",
    ]

    if _has_files(expected_files):
        print("WikiText-103 already exists locally. Skipping download.")
        return

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'datasets' package is required to download WikiText-103. "
            "Install it with: python3 -m pip install --user datasets"
        ) from exc

    print("Downloading WikiText-103 from HuggingFace datasets...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    wikitext_dir.mkdir(parents=True, exist_ok=True)

    split_to_file = {
        "train": wikitext_dir / "wiki.train.tokens",
        "validation": wikitext_dir / "wiki.valid.tokens",
        "test": wikitext_dir / "wiki.test.tokens",
    }

    for split_name, out_path in split_to_file.items():
        with out_path.open("w", encoding="utf-8") as fout:
            for row in tqdm(dataset[split_name], desc=f"Writing {split_name}"):
                text = row.get("text", "")
                fout.write(text + "\n")


def ensure_text8(raw_data_dir: Path) -> None:
    text8_dir = raw_data_dir / "text8"
    text8_file = text8_dir / "text8"

    if text8_file.exists():
        print("Text8 already exists locally. Skipping download.")
        return

    archive_path = raw_data_dir / "text8.zip"
    if not archive_path.exists():
        print("Downloading Text8...")
        _download_with_progress(TEXT8_URL, archive_path)

    text8_dir.mkdir(parents=True, exist_ok=True)
    _extract_zip(archive_path, text8_dir)


def ensure_enwik8(raw_data_dir: Path) -> None:
    enwik8_dir = raw_data_dir / "enwik8"
    enwik8_file = enwik8_dir / "enwik8"

    if enwik8_file.exists():
        print("Enwik8 already exists locally. Skipping download.")
        return

    archive_path = raw_data_dir / "enwik8.zip"
    if not archive_path.exists():
        print("Downloading Enwik8...")
        _download_with_progress(ENWIK8_URL, archive_path)

    enwik8_dir.mkdir(parents=True, exist_ok=True)
    _extract_zip(archive_path, enwik8_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and validate datasets for tokenization assignment."
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("../data"),
        help="Directory where raw datasets are stored or downloaded.",
    )
    parser.add_argument(
        "--skip-one-billion",
        action="store_true",
        help="Skip One Billion Word Benchmark setup.",
    )
    parser.add_argument(
        "--skip-wikitext",
        action="store_true",
        help="Skip WikiText-103 setup.",
    )
    parser.add_argument(
        "--skip-text8",
        action="store_true",
        help="Skip Text8 setup.",
    )
    parser.add_argument(
        "--skip-enwik8",
        action="store_true",
        help="Skip Enwik8 setup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_data_dir = args.raw_data_dir.resolve()
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_one_billion:
        ensure_one_billion(raw_data_dir)
    if not args.skip_wikitext:
        ensure_wikitext(raw_data_dir)
    if not args.skip_text8:
        ensure_text8(raw_data_dir)
    if not args.skip_enwik8:
        ensure_enwik8(raw_data_dir)

    print("Data setup complete.")


if __name__ == "__main__":
    main()
