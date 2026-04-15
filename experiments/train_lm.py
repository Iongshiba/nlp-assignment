from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.lm_model import LSTMLanguageModel
from tokenizers_impl.bpe_tokenizer import BPETokenizerConfig, BPETokenizerWrapper
from tokenizers_impl.char_tokenizer import CharTokenizer, CharTokenizerConfig
from tokenizers_impl.word_tokenizer import WordTokenizer, WordTokenizerConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split(
    processed_dir: Path,
    dataset_name: str,
    split_name: str,
    max_samples: int | None = None,
) -> List[str]:
    split_path = processed_dir / dataset_name / f"{split_name}.txt"
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


def build_tokenizer(args: argparse.Namespace, train_samples: List[str]):
    if args.tokenizer == "word":
        tokenizer = WordTokenizer(WordTokenizerConfig(vocab_size=args.word_vocab_size))
    elif args.tokenizer == "char":
        tokenizer = CharTokenizer(CharTokenizerConfig(max_vocab_size=args.char_vocab_size))
    elif args.tokenizer == "bpe":
        tokenizer = BPETokenizerWrapper(
            BPETokenizerConfig(vocab_size=args.bpe_vocab_size)
        )
    else:
        raise ValueError(f"Unsupported tokenizer: {args.tokenizer}")

    tokenizer.train(train_samples)
    return tokenizer


def flatten_token_ids(samples: List[str], tokenizer) -> np.ndarray:
    token_ids: List[int] = []
    eos_id = tokenizer.eos_id

    for sample in samples:
        encoded = tokenizer.encode(sample)
        if encoded:
            token_ids.extend(encoded)
            token_ids.append(eos_id)

    if len(token_ids) < 2:
        raise RuntimeError("Not enough tokenized data to train language model.")

    return np.asarray(token_ids, dtype=np.int64)


def iter_lm_batches(
    token_ids: np.ndarray, batch_size: int, sequence_length: int
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    total_tokens = len(token_ids) - 1
    usable_tokens = (total_tokens // batch_size) * batch_size

    if usable_tokens <= batch_size:
        raise RuntimeError("Token sequence is too short for the selected batch size.")

    x_data = token_ids[:usable_tokens].reshape(batch_size, -1)
    y_data = token_ids[1 : usable_tokens + 1].reshape(batch_size, -1)

    steps_per_batch = x_data.shape[1] // sequence_length
    for step in range(steps_per_batch):
        start = step * sequence_length
        end = start + sequence_length
        x_batch = torch.from_numpy(x_data[:, start:end].copy()).long()
        y_batch = torch.from_numpy(y_data[:, start:end].copy()).long()
        yield x_batch, y_batch


def evaluate_perplexity(
    model: nn.Module,
    token_ids: np.ndarray,
    criterion: nn.Module,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for x_batch, y_batch in iter_lm_batches(token_ids, batch_size, sequence_length):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits, _ = model(x_batch)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y_batch.reshape(-1))
            total_loss += loss.item()
            total_batches += 1

    if total_batches == 0:
        return float("inf")

    avg_loss = total_loss / total_batches
    return math.exp(min(avg_loss, 20.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an LSTM language model with selectable tokenization."
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("artifacts/processed")
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext103",
        choices=["wikitext103", "text8", "one_billion", "enwik8"],
    )
    parser.add_argument(
        "--tokenizer", type=str, default="bpe", choices=["word", "char", "bpe"]
    )
    parser.add_argument("--word-vocab-size", type=int, default=50000)
    parser.add_argument("--char-vocab-size", type=int, default=None)
    parser.add_argument("--bpe-vocab-size", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/lm"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    processed_dir = args.processed_dir.resolve()
    train_samples = read_split(
        processed_dir, args.dataset, "train", max_samples=args.max_train_samples
    )
    valid_samples = read_split(
        processed_dir,
        args.dataset,
        "validation",
        max_samples=args.max_validation_samples,
    )

    print(
        f"Loaded {len(train_samples)} training samples and {len(valid_samples)} validation samples."
    )
    print(f"Training tokenizer: {args.tokenizer}")

    tokenizer = build_tokenizer(args, train_samples)
    train_token_ids = flatten_token_ids(train_samples, tokenizer)
    valid_token_ids = flatten_token_ids(valid_samples, tokenizer)

    vocab_size = tokenizer.vocab_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    if args.tokenizer == "word":
        run_name = f"word_{args.word_vocab_size}"
    elif args.tokenizer == "char":
        run_name = (
            f"char_{args.char_vocab_size}"
            if args.char_vocab_size is not None
            else "char_full"
        )
    else:
        run_name = f"bpe_{args.bpe_vocab_size}"
    run_dir = args.output_dir.resolve() / args.dataset / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = run_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    metrics_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        total_loss = 0.0
        total_batches = 0

        train_batches = iter_lm_batches(
            train_token_ids, args.batch_size, args.sequence_length
        )
        for x_batch, y_batch in tqdm(
            train_batches, desc=f"Epoch {epoch}/{args.epochs}"
        ):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits, _ = model(x_batch)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y_batch.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        epoch_time = time.perf_counter() - epoch_start
        avg_train_loss = total_loss / max(total_batches, 1)
        train_perplexity = math.exp(min(avg_train_loss, 20.0))

        val_perplexity = evaluate_perplexity(
            model=model,
            token_ids=valid_token_ids,
            criterion=criterion,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_perplexity": train_perplexity,
            "validation_perplexity": val_perplexity,
            "epoch_time_seconds": epoch_time,
            "vocab_size": vocab_size,
            "dataset": args.dataset,
            "tokenizer": run_name,
        }
        metrics_rows.append(row)

        print(
            f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
            f"train_ppl={train_perplexity:.2f}, val_ppl={val_perplexity:.2f}, "
            f"time={epoch_time:.2f}s"
        )

    model_path = run_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    print(f"Saved tokenizer to {tokenizer_path}")
    print(f"Saved model to {model_path}")
    print(f"Saved LM metrics to {metrics_path}")


if __name__ == "__main__":
    main()
