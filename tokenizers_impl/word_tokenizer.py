from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


@dataclass
class WordTokenizerConfig:
    vocab_size: int = 50000


class WordTokenizer:
    def __init__(self, config: WordTokenizerConfig | None = None) -> None:
        self.config = config or WordTokenizerConfig()
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return TOKEN_PATTERN.findall(text)

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<UNK>"]

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    def train(self, texts: List[str]) -> None:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self.tokenize(text))

        remaining_slots = max(0, self.config.vocab_size - len(SPECIAL_TOKENS))
        sorted_tokens = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        learned_tokens = [token for token, _ in sorted_tokens[:remaining_slots]]

        self.id_to_token = SPECIAL_TOKENS + learned_tokens
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> List[int]:
        if not self.token_to_id:
            raise RuntimeError("Tokenizer has not been trained.")

        token_ids: List[int] = []
        if add_bos:
            token_ids.append(self.bos_id)

        for token in self.tokenize(text):
            token_ids.append(self.token_to_id.get(token, self.unk_id))

        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        tokens = [
            self.id_to_token[idx] for idx in token_ids if idx < len(self.id_to_token)
        ]
        return " ".join(tokens)

    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def save(self, output_path: Path) -> None:
        payload = {
            "config": {"vocab_size": self.config.vocab_size},
            "id_to_token": self.id_to_token,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fout:
            json.dump(payload, fout, indent=2)

    @classmethod
    def load(cls, input_path: Path) -> "WordTokenizer":
        with input_path.open("r", encoding="utf-8") as fin:
            payload = json.load(fin)

        config = WordTokenizerConfig(vocab_size=payload["config"]["vocab_size"])
        tokenizer = cls(config=config)
        tokenizer.id_to_token = payload["id_to_token"]
        tokenizer.token_to_id = {
            token: idx for idx, token in enumerate(tokenizer.id_to_token)
        }
        return tokenizer


def compute_word_metrics(
    tokenizer: WordTokenizer, samples: List[str], include_oov: bool = True
) -> Dict[str, float]:
    sequence_lengths: List[int] = []
    total_tokens = 0
    total_chars = 0
    oov_tokens = 0

    for sample in samples:
        tokens = tokenizer.tokenize(sample)
        sequence_lengths.append(len(tokens))
        total_tokens += len(tokens)
        total_chars += len(sample)

        if include_oov:
            for token in tokens:
                if token not in tokenizer.token_to_id:
                    oov_tokens += 1

    avg_sequence_length = (
        (sum(sequence_lengths) / len(sequence_lengths)) if sequence_lengths else 0.0
    )
    oov_rate = (
        (oov_tokens / total_tokens * 100.0) if include_oov and total_tokens > 0 else 0.0
    )
    compression_ratio = (total_tokens / total_chars) if total_chars > 0 else 0.0

    return {
        "vocab_size": float(tokenizer.vocab_size()),
        "avg_sequence_length": avg_sequence_length,
        "oov_rate": oov_rate,
        "compression_ratio": compression_ratio,
    }


def _read_split(file_path: Path) -> List[str]:
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as fin:
        return [line.strip() for line in fin if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a word-level tokenizer."
    )
    parser.add_argument("--processed-dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--vocab-sizes", type=int, nargs="+", default=[2000, 8000, 16000, 32000, 50000],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_samples = _read_split(args.processed_dataset_dir / "train.txt")
    valid_samples = _read_split(args.processed_dataset_dir / "validation.txt")
    test_samples = _read_split(args.processed_dataset_dir / "test.txt")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for vocab_size in args.vocab_sizes:
        print(f"Training word tokenizer with vocab size = {vocab_size}")
        experiment_start = time.perf_counter()
        tokenizer = WordTokenizer(WordTokenizerConfig(vocab_size=vocab_size))
        tokenizer.train(train_samples)

        tokenizer_out = args.output_dir / f"word_tokenizer_{vocab_size}.json"
        tokenizer.save(tokenizer_out)

        metrics = {
            "train": compute_word_metrics(tokenizer, train_samples, True),
            "validation": compute_word_metrics(tokenizer, valid_samples, True),
            "test": compute_word_metrics(tokenizer, test_samples, True),
            "elapsed_time_seconds": time.perf_counter() - experiment_start,
        }

        metrics_out = args.output_dir / f"word_metrics_{vocab_size}.json"
        with metrics_out.open("w", encoding="utf-8") as fout:
            json.dump(metrics, fout, indent=2)

        print(f"Saved tokenizer to {tokenizer_out}")
        print(f"Saved metrics to {metrics_out}")


if __name__ == "__main__":
    main()
