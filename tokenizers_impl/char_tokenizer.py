from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


@dataclass
class CharTokenizerConfig:
    max_vocab_size: int | None = None


class CharTokenizer:
    def __init__(self, config: CharTokenizerConfig | None = None) -> None:
        self.config = config or CharTokenizerConfig()
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<UNK>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    def train(self, texts: List[str]) -> None:
        charset = sorted(set("".join(texts)))
        if self.config.max_vocab_size is not None:
            limit = max(0, self.config.max_vocab_size - len(SPECIAL_TOKENS))
            charset = charset[:limit]

        self.id_to_token = SPECIAL_TOKENS + charset
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> List[int]:
        if not self.token_to_id:
            raise RuntimeError("Tokenizer has not been trained.")

        token_ids: List[int] = []
        if add_bos:
            token_ids.append(self.bos_id)

        for ch in text:
            token_ids.append(self.token_to_id.get(ch, self.unk_id))

        if add_eos:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        chars = [
            self.id_to_token[idx] for idx in token_ids if idx < len(self.id_to_token)
        ]
        return "".join(chars)

    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def save(self, output_path: Path) -> None:
        payload = {
            "config": {"max_vocab_size": self.config.max_vocab_size},
            "id_to_token": self.id_to_token,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fout:
            json.dump(payload, fout, indent=2)

    @classmethod
    def load(cls, input_path: Path) -> "CharTokenizer":
        with input_path.open("r", encoding="utf-8") as fin:
            payload = json.load(fin)

        config = CharTokenizerConfig(max_vocab_size=payload["config"]["max_vocab_size"])
        tokenizer = cls(config=config)
        tokenizer.id_to_token = payload["id_to_token"]
        tokenizer.token_to_id = {
            token: idx for idx, token in enumerate(tokenizer.id_to_token)
        }
        return tokenizer


def compute_char_metrics(
    tokenizer: CharTokenizer, samples: List[str]
) -> Dict[str, float]:
    sequence_lengths: List[int] = []
    total_tokens = 0
    total_chars = 0

    for sample in samples:
        length = len(sample)
        sequence_lengths.append(length)
        total_tokens += length
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
        return [line.rstrip("\n") for line in fin if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a character-level tokenizer."
    )
    parser.add_argument("--processed-dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-vocab-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_samples = _read_split(args.processed_dataset_dir / "train.txt")
    valid_samples = _read_split(args.processed_dataset_dir / "validation.txt")
    test_samples = _read_split(args.processed_dataset_dir / "test.txt")

    tokenizer = CharTokenizer(CharTokenizerConfig(max_vocab_size=args.max_vocab_size))
    tokenizer.train(train_samples)

    tokenizer_out = args.output_dir / "char_tokenizer.json"
    tokenizer.save(tokenizer_out)

    metrics = {
        "train": compute_char_metrics(tokenizer, train_samples),
        "validation": compute_char_metrics(tokenizer, valid_samples),
        "test": compute_char_metrics(tokenizer, test_samples),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_out = args.output_dir / "char_metrics.json"
    with metrics_out.open("w", encoding="utf-8") as fout:
        json.dump(metrics, fout, indent=2)

    print(f"Saved tokenizer to {tokenizer_out}")
    print(f"Saved metrics to {metrics_out}")


if __name__ == "__main__":
    main()
