from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


@dataclass
class BPETokenizerConfig:
    vocab_size: int


class BPETokenizerWrapper:
    def __init__(self, config: BPETokenizerConfig) -> None:
        self.config = config
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = Whitespace()

    @property
    def pad_id(self) -> int:
        token_id = self.tokenizer.token_to_id("<PAD>")
        if token_id is None:
            raise RuntimeError("Tokenizer has not been trained.")
        return token_id

    @property
    def unk_id(self) -> int:
        token_id = self.tokenizer.token_to_id("<UNK>")
        if token_id is None:
            raise RuntimeError("Tokenizer has not been trained.")
        return token_id

    @property
    def bos_id(self) -> int:
        token_id = self.tokenizer.token_to_id("<BOS>")
        if token_id is None:
            raise RuntimeError("Tokenizer has not been trained.")
        return token_id

    @property
    def eos_id(self) -> int:
        token_id = self.tokenizer.token_to_id("<EOS>")
        if token_id is None:
            raise RuntimeError("Tokenizer has not been trained.")
        return token_id

    def train(self, texts: List[str]) -> None:
        trainer = BpeTrainer(
            vocab_size=self.config.vocab_size, special_tokens=SPECIAL_TOKENS
        )
        self.tokenizer.train_from_iterator(texts, trainer=trainer)

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> List[int]:
        encoded = self.tokenizer.encode(text)
        token_ids = list(encoded.ids)
        if add_bos:
            token_ids = [self.bos_id] + token_ids
        if add_eos:
            token_ids = token_ids + [self.eos_id]
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(output_path))

    @classmethod
    def load(cls, input_path: Path) -> "BPETokenizerWrapper":
        tokenizer = Tokenizer.from_file(str(input_path))
        wrapper = cls(BPETokenizerConfig(vocab_size=tokenizer.get_vocab_size()))
        wrapper.tokenizer = tokenizer
        return wrapper


def compute_bpe_metrics(
    tokenizer: BPETokenizerWrapper, samples: List[str]
) -> Dict[str, float]:
    sequence_lengths: List[int] = []
    total_tokens = 0
    total_chars = 0

    for sample in samples:
        token_ids = tokenizer.encode(sample)
        sequence_lengths.append(len(token_ids))
        total_tokens += len(token_ids)
        total_chars += len(sample)

    avg_sequence_length = (
        (sum(sequence_lengths) / len(sequence_lengths)) if sequence_lengths else 0.0
    )
    compression_ratio = (total_tokens / total_chars) if total_chars > 0 else 0.0

    return {
        "vocab_size": float(tokenizer.vocab_size()),
        "avg_sequence_length": avg_sequence_length,
        "compression_ratio": compression_ratio,
    }


def _read_split(file_path: Path) -> List[str]:
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as fin:
        return [line.strip() for line in fin if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate BPE tokenizers.")
    parser.add_argument("--processed-dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--vocab-sizes", type=int, nargs="+", default=[1000, 5000, 10000, 30000]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_samples = _read_split(args.processed_dataset_dir / "train.txt")
    valid_samples = _read_split(args.processed_dataset_dir / "validation.txt")
    test_samples = _read_split(args.processed_dataset_dir / "test.txt")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for vocab_size in args.vocab_sizes:
        print(f"Training BPE tokenizer with vocab size = {vocab_size}")
        experiment_start = time.perf_counter()
        tokenizer = BPETokenizerWrapper(BPETokenizerConfig(vocab_size=vocab_size))
        tokenizer.train(train_samples)

        tokenizer_out = args.output_dir / f"bpe_tokenizer_{vocab_size}.json"
        tokenizer.save(tokenizer_out)

        metrics = {
            "train": compute_bpe_metrics(tokenizer, train_samples),
            "validation": compute_bpe_metrics(tokenizer, valid_samples),
            "test": compute_bpe_metrics(tokenizer, test_samples),
            "elapsed_time_seconds": time.perf_counter() - experiment_start,
        }

        metrics_out = args.output_dir / f"bpe_metrics_{vocab_size}.json"
        with metrics_out.open("w", encoding="utf-8") as fout:
            json.dump(metrics, fout, indent=2)

        print(f"Saved tokenizer to {tokenizer_out}")
        print(f"Saved metrics to {metrics_out}")


if __name__ == "__main__":
    main()
