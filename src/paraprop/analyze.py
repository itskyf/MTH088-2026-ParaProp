"""Analyze and compare SGD vs QuickProp experiment results from Trackio.

Fetches metrics via the Trackio CLI, computes comparison tables (mean ±
std across seeds), and generates publication-ready matplotlib figures.
"""

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT = "paraprop"
SEEDS = [42, 3407, 1337]
OUTPUT_DIR = Path("outputs")

# Metrics to fetch from Trackio
METRIC_KEYS = [
    "train/loss",
    "train/MulticlassAccuracy",
    "train/MulticlassF1Score",
    "val/MulticlassAccuracy",
    "val/MulticlassF1Score",
]

# Two experimental regimes (do NOT mix on same figure)
REGIMES: dict[str, dict] = {
    "fullbatch": {
        "num_epochs": 150,
        "runs": {
            "SGD": ("optimizer:SGD-minibatch:False-grad_clip:off-seed:{seed}-lr:0.03"),
            "QuickProp": (
                "optimizer:QuickProp-minibatch:False-grad_clip:off-seed:{seed}-lr:0.03"
            ),
        },
    },
    "minibatch": {
        "num_epochs": 30,
        "runs": {
            "SGD": ("optimizer:SGD-minibatch:True-grad_clip:1.0-seed:{seed}-lr:0.1"),
            "QuickProp": (
                "optimizer:QuickProp-minibatch:True-grad_clip:1.0-seed:{seed}-lr:0.01"
            ),
        },
    },
}

# Performance thresholds for "steps to target" analysis
ACC_THRESHOLD = 0.75
F1_THRESHOLD = 0.75


# ── Data fetching ────────────────────────────────────────────────────────────


def fetch_metric(run: str, metric: str) -> list[dict]:
    """Fetch metric values from Trackio CLI as a list of {step, value}
    dicts."""
    result = subprocess.run(
        [
            "uv",
            "run",
            "trackio",
            "get",
            "metric",
            "--project",
            PROJECT,
            "--run",
            run,
            "--metric",
            metric,
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)["values"]


def fetch_all_metrics() -> pl.DataFrame:
    """Fetch all metrics for every run and return a single tall DataFrame.

    Columns: regime, optimizer, seed, epoch, metric, value
    """
    rows: list[dict] = []
    for regime_name, regime_cfg in REGIMES.items():
        for opt_name, run_tpl in regime_cfg["runs"].items():
            for seed in SEEDS:
                run = run_tpl.format(seed=seed)
                for metric_key in METRIC_KEYS:
                    values = fetch_metric(run, metric_key)
                    rows.extend(
                        {
                            "regime": regime_name,
                            "optimizer": opt_name,
                            "seed": seed,
                            "epoch": v["step"],
                            "metric": metric_key,
                            "value": v["value"],
                        }
                        for v in values
                    )
    return pl.DataFrame(rows)


# ── Analysis helpers ─────────────────────────────────────────────────────────


def compute_summary_table(df: pl.DataFrame) -> pl.DataFrame:
    """Compute mean ± std of final-epoch metrics across seeds.

    Returns: regime, optimizer, metric, mean, std, formatted
    """
    # Get the last epoch per (regime, optimizer, seed, metric)
    final = df.group_by("regime", "optimizer", "seed", "metric").agg(
        pl.col("value").sort_by("epoch").last().alias("final_value")
    )
    summary = (
        final.group_by("regime", "optimizer", "metric")
        .agg(
            pl.col("final_value").mean().alias("mean"),
            pl.col("final_value").std().alias("std"),
        )
        .with_columns(
            (
                pl.col("mean").round(4).cast(pl.Utf8)
                + " ± "
                + pl.col("std").round(4).cast(pl.Utf8)
            ).alias("formatted")
        )
        .sort("regime", "metric", "optimizer")
    )
    return summary


def compute_steps_to_target(
    df: pl.DataFrame,
    metric: str,
    threshold: float,
) -> pl.DataFrame:
    """Find the first epoch where each run crosses *threshold*.

    Returns: regime, optimizer, seed, first_epoch (null if never reached)
    """
    filtered = df.filter((pl.col("metric") == metric) & (pl.col("value") >= threshold))
    first_hit = filtered.group_by("regime", "optimizer", "seed").agg(
        pl.col("epoch").min().alias("first_epoch")
    )
    # Ensure all (regime, optimizer, seed) combos are present
    all_combos = (
        df.filter(pl.col("metric") == metric)
        .select("regime", "optimizer", "seed")
        .unique()
    )
    result = all_combos.join(first_hit, on=["regime", "optimizer", "seed"], how="left")
    return result


def compute_steps_to_target_summary(
    df: pl.DataFrame,
    metric: str,
    threshold: float,
) -> pl.DataFrame:
    """Mean ± std of first epoch reaching *threshold*, grouped by regime &
    optimizer."""
    steps = compute_steps_to_target(df, metric, threshold)
    summary = (
        steps.group_by("regime", "optimizer")
        .agg(
            pl.col("first_epoch").mean().alias("mean_epoch"),
            pl.col("first_epoch").std().alias("std_epoch"),
            pl.col("first_epoch").is_null().sum().alias("n_never_reached"),
        )
        .sort("regime", "optimizer")
    )
    return summary


def compute_loss_auc(df: pl.DataFrame) -> pl.DataFrame:
    """Compute trapezoidal AUC of the train/loss curve per run.

    Returns: regime, optimizer, seed, loss_auc
    """
    loss = df.filter(pl.col("metric") == "train/loss").sort(
        "regime", "optimizer", "seed", "epoch"
    )

    def _trapz(group: pl.DataFrame) -> float:
        epochs = group["epoch"].to_numpy().astype(float)
        values = group["value"].to_numpy().astype(float)
        return float(np.trapezoid(values, epochs))

    rows: list[dict] = []
    for (regime, opt, seed), group in loss.group_by(
        "regime", "optimizer", "seed", maintain_order=True
    ):
        rows.append(
            {
                "regime": regime,
                "optimizer": opt,
                "seed": seed,
                "loss_auc": _trapz(group),
            }
        )
    return pl.DataFrame(rows)


def compute_loss_auc_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Mean ± std of loss AUC across seeds."""
    auc = compute_loss_auc(df)
    summary = (
        auc.group_by("regime", "optimizer")
        .agg(
            pl.col("loss_auc").mean().alias("mean_auc"),
            pl.col("loss_auc").std().alias("std_auc"),
        )
        .sort("regime", "optimizer")
    )
    return summary


# ── Plotting helpers ─────────────────────────────────────────────────────────

# Style constants
COLORS = {"SGD": "#2563eb", "QuickProp": "#e11d48"}
STYLE = {"SGD": "-", "QuickProp": "--"}


def _setup_style():
    """Apply a polished dark matplotlib style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "#181825",
            "axes.facecolor": "#1e1e2e",
            "axes.edgecolor": "#45475a",
            "axes.labelcolor": "#cdd6f4",
            "axes.grid": True,
            "grid.color": "#313244",
            "grid.alpha": 0.6,
            "text.color": "#cdd6f4",
            "xtick.color": "#a6adc8",
            "ytick.color": "#a6adc8",
            "legend.facecolor": "#1e1e2e",
            "legend.edgecolor": "#45475a",
            "legend.fontsize": 9,
            "font.family": "sans-serif",
            "font.size": 11,
        }
    )


def _mean_std_by_epoch(
    df: pl.DataFrame,
    regime: str,
    optimizer: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (epochs, mean_values, std_values) across seeds."""
    sub = df.filter(
        (pl.col("regime") == regime)
        & (pl.col("optimizer") == optimizer)
        & (pl.col("metric") == metric)
    )
    agg = (
        sub.group_by("epoch")
        .agg(
            pl.col("value").mean().alias("mean"),
            pl.col("value").std().alias("std"),
        )
        .sort("epoch")
    )
    epochs = agg["epoch"].to_numpy()
    means = agg["mean"].to_numpy()
    stds = agg["std"].to_numpy()
    return epochs, means, stds


def plot_metric_curves(
    df: pl.DataFrame,
    regime: str,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
):
    """Plot mean±std curves for SGD vs QuickProp."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for opt in ("SGD", "QuickProp"):
        epochs, means, stds = _mean_std_by_epoch(
            df,
            regime,
            opt,
            metric,
        )
        ax.plot(
            epochs,
            means,
            STYLE[opt],
            color=COLORS[opt],
            label=opt,
            linewidth=2,
        )
        ax.fill_between(
            epochs,
            means - stds,
            means + stds,
            color=COLORS[opt],
            alpha=0.15,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_bar_comparison(
    summary_df: pl.DataFrame,
    value_col: str,
    std_col: str,
    ylabel: str,
    title: str,
    filename: str,
):
    """Grouped bar chart comparing SGD vs QuickProp across regimes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    regimes = summary_df["regime"].unique().sort().to_list()
    optimizers = ["SGD", "QuickProp"]
    x = np.arange(len(regimes))
    width = 0.3

    for i, opt in enumerate(optimizers):
        sub = summary_df.filter(pl.col("optimizer") == opt).sort("regime")
        means = sub[value_col].to_numpy()
        stds = sub[std_col].to_numpy()
        # Replace NaN stds with 0 for plotting
        stds = np.nan_to_num(stds, nan=0.0)
        bars = ax.bar(
            x + (i - 0.5) * width,
            means,
            width,
            yerr=stds,
            label=opt,
            color=COLORS[opt],
            edgecolor="white",
            linewidth=0.5,
            capsize=4,
            alpha=0.85,
        )
        # Value labels on bars
        for bar, m, s in zip(bars, means, stds, strict=True):
            if not np.isnan(m):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.01 * abs(ax.get_ylim()[1]),
                    f"{m:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#cdd6f4",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("batch", "-batch") for r in regimes])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_steps_to_target_bar(
    steps_df: pl.DataFrame,
    threshold: float,
    metric_label: str,
    filename: str,
):
    """Bar chart of mean epochs to reach a threshold."""
    # Mark "never reached" with the max epoch of the regime + some offset
    display = steps_df.clone()
    fig, ax = plt.subplots(figsize=(8, 5))
    regimes = display["regime"].unique().sort().to_list()
    optimizers = ["SGD", "QuickProp"]
    x = np.arange(len(regimes))
    width = 0.3

    for i, opt in enumerate(optimizers):
        sub = display.filter(pl.col("optimizer") == opt).sort("regime")
        means = sub["mean_epoch"].to_numpy()
        stds = sub["std_epoch"].to_numpy()
        n_never = sub["n_never_reached"].to_numpy()
        stds = np.nan_to_num(stds, nan=0.0)
        means_plot = np.nan_to_num(means, nan=0.0)

        bars = ax.bar(
            x + (i - 0.5) * width,
            means_plot,
            width,
            yerr=stds,
            label=opt,
            color=COLORS[opt],
            edgecolor="white",
            linewidth=0.5,
            capsize=4,
            alpha=0.85,
        )
        for bar, m, s, nn_ in zip(bars, means, stds, n_never, strict=True):
            label_text = f"{m:.1f}" if not np.isnan(m) else "N/A"
            if nn_ > 0:
                label_text += f"\n({nn_}/3 missed)"
            y_pos = bar.get_height() + s + 0.5
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                label_text,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#cdd6f4",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("batch", "-batch") for r in regimes])
    ax.set_ylabel("Epochs to target")
    ax.set_title(
        f"Epochs to reach {threshold:.0%} {metric_label} (mean ± std, 3 seeds)",
        fontweight="bold",
        fontsize=12,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


# ── Main entry point ─────────────────────────────────────────────────────────


def main():
    """Run the full analysis pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _setup_style()

    # 1. Fetch all data
    print("Fetching metrics from Trackio...")
    df = fetch_all_metrics()
    print(f"  Collected {len(df)} data points.\n")

    # 2. Final-metric summary table (mean ± std)
    print("═" * 60)
    print("FINAL-EPOCH METRICS (mean ± std across 3 seeds)")
    print("═" * 60)
    summary = compute_summary_table(df)
    for regime in ("fullbatch", "minibatch"):
        regime_label = (
            "Full-batch (150 epochs)"
            if regime == "fullbatch"
            else "Mini-batch (30 epochs)"
        )
        print(f"\n▸ {regime_label}")
        sub = summary.filter(
            pl.col("regime") == regime,
        ).select("optimizer", "metric", "formatted")
        print(sub)

    # 3. Steps to target
    print("\n" + "═" * 60)
    print("STEPS/EPOCHS TO TARGET")
    print("═" * 60)

    acc_target = compute_steps_to_target_summary(
        df,
        "val/MulticlassAccuracy",
        ACC_THRESHOLD,
    )
    f1_target = compute_steps_to_target_summary(
        df,
        "val/MulticlassF1Score",
        F1_THRESHOLD,
    )

    print(f"\n▸ Epochs to {ACC_THRESHOLD:.0%} val accuracy:")
    print(acc_target)
    print(f"\n▸ Epochs to {F1_THRESHOLD:.0%} val F1:")
    print(f1_target)

    # 4. Loss AUC
    print("\n" + "═" * 60)
    print("LOSS AUC (lower = faster convergence)")
    print("═" * 60)
    auc_summary = compute_loss_auc_summary(df)
    print(auc_summary)

    # ── Generate figures ─────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("GENERATING FIGURES")
    print("═" * 60)

    # Per-regime metric curves
    for regime in ("fullbatch", "minibatch"):
        n_ep = REGIMES[regime]["num_epochs"]
        label = f"{'Full' if regime == 'fullbatch' else 'Mini'}-batch ({n_ep} epochs)"

        plot_metric_curves(
            df,
            regime,
            "train/loss",
            ylabel="Training Loss",
            title=f"Training Loss — {label}",
            filename=f"{regime}_train_loss.png",
        )
        plot_metric_curves(
            df,
            regime,
            "val/MulticlassAccuracy",
            ylabel="Validation Accuracy",
            title=f"Validation Accuracy — {label}",
            filename=f"{regime}_val_accuracy.png",
        )
        plot_metric_curves(
            df,
            regime,
            "val/MulticlassF1Score",
            ylabel="Validation F1 (macro)",
            title=f"Validation F1 (macro) — {label}",
            filename=f"{regime}_val_f1.png",
        )
        plot_metric_curves(
            df,
            regime,
            "train/MulticlassAccuracy",
            ylabel="Training Accuracy",
            title=f"Training Accuracy — {label}",
            filename=f"{regime}_train_accuracy.png",
        )

    # Bar charts
    plot_bar_comparison(
        auc_summary,
        "mean_auc",
        "std_auc",
        ylabel="Loss AUC",
        title="Training Loss AUC — SGD vs QuickProp (mean ± std, 3 seeds)",
        filename="bar_loss_auc.png",
    )
    plot_steps_to_target_bar(
        acc_target,
        ACC_THRESHOLD,
        "Accuracy",
        filename="bar_epochs_to_75pct_accuracy.png",
    )
    plot_steps_to_target_bar(
        f1_target,
        F1_THRESHOLD,
        "F1",
        filename="bar_epochs_to_75pct_f1.png",
    )

    print("\n✅ Analysis complete. All figures saved to outputs/")


if __name__ == "__main__":
    main()
