import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PALETTE = {
    "word": "#4C72B0",
    "bpe": "#DD8452",
    "char": "#55A868",
}


def build_dataframe() -> pd.DataFrame:
    data = [
        {"tokenizer_run": "word_2000", "effective_vocab": 2000, "final_validation_ppl": 20.853, "avg_epoch_time_s": 107.805},
        {"tokenizer_run": "word_36000", "effective_vocab": 36000, "final_validation_ppl": 91.595, "avg_epoch_time_s": 247.414},
        {"tokenizer_run": "bpe_2000", "effective_vocab": 2000, "final_validation_ppl": 53.018, "avg_epoch_time_s": 540.036},
        {"tokenizer_run": "bpe_36000", "effective_vocab": 36000, "final_validation_ppl": 178.612, "avg_epoch_time_s": 255.976},
        {"tokenizer_run": "char_2000", "effective_vocab": 52, "final_validation_ppl": 3.786, "avg_epoch_time_s": 345.279},
    ]
    df = pd.DataFrame(data)
    df["tokenizer_family"] = df["tokenizer_run"].str.split("_").str[0]
    return df


def annotate_bars(ax: plt.Axes, fmt: str = "{:.1f}") -> None:
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            fmt.format(height),
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )


def plot_perplexity(df: pd.DataFrame, out_dir: str) -> None:
    order = df.sort_values("final_validation_ppl")["tokenizer_run"].tolist()

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=df,
        x="tokenizer_run",
        y="final_validation_ppl",
        hue="tokenizer_family",
        order=order,
        dodge=False,
        palette=PALETTE,
    )
    annotate_bars(ax, "{:.3f}")
    ax.set_title("Enwik8: Final Validation Perplexity by Tokenizer Run")
    ax.set_xlabel("Tokenizer Run")
    ax.set_ylabel("Final Validation PPL (lower is better)")
    ax.legend(title="Tokenizer Family")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "enwik8_lm_perplexity_bar.png"), dpi=200)
    plt.close()


def plot_speed(df: pd.DataFrame, out_dir: str) -> None:
    order = df.sort_values("avg_epoch_time_s")["tokenizer_run"].tolist()

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=df,
        x="tokenizer_run",
        y="avg_epoch_time_s",
        hue="tokenizer_family",
        order=order,
        dodge=False,
        palette=PALETTE,
    )
    annotate_bars(ax, "{:.1f}")
    ax.set_title("Enwik8: Average Epoch Training Time by Tokenizer Run")
    ax.set_xlabel("Tokenizer Run")
    ax.set_ylabel("Avg Epoch Time (s, lower is better)")
    ax.legend(title="Tokenizer Family")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "enwik8_lm_speed_bar.png"), dpi=200)
    plt.close()


def plot_tradeoff(df: pd.DataFrame, out_dir: str) -> None:
    plt.figure(figsize=(8.5, 6))
    ax = sns.scatterplot(
        data=df,
        x="avg_epoch_time_s",
        y="final_validation_ppl",
        hue="tokenizer_family",
        style="tokenizer_family",
        palette=PALETTE,
        s=170,
    )

    for _, row in df.iterrows():
        ax.text(
            row["avg_epoch_time_s"] + 8,
            row["final_validation_ppl"] + 2,
            row["tokenizer_run"],
            fontsize=9,
        )

    ax.set_title("Enwik8: Perplexity vs Training Speed Trade-off")
    ax.set_xlabel("Avg Epoch Time (s, lower is better)")
    ax.set_ylabel("Final Validation PPL (lower is better)")
    ax.legend(title="Tokenizer Family")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "enwik8_lm_tradeoff_scatter.png"), dpi=200)
    plt.close()


def print_summary(df: pd.DataFrame) -> None:
    best_ppl = df.loc[df["final_validation_ppl"].idxmin()]
    best_speed = df.loc[df["avg_epoch_time_s"].idxmin()]

    print("Best perplexity:", best_ppl["tokenizer_run"], f"({best_ppl['final_validation_ppl']:.3f})")
    print("Best speed:", best_speed["tokenizer_run"], f"({best_speed['avg_epoch_time_s']:.3f} s/epoch)")


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root_dir, "report", "enwik8", "final_analysis", "plots")
    os.makedirs(out_dir, exist_ok=True)

    df = build_dataframe()

    plot_perplexity(df, out_dir)
    plot_speed(df, out_dir)
    plot_tradeoff(df, out_dir)
    print_summary(df)

    print("Saved charts to:", out_dir)


if __name__ == "__main__":
    main()
