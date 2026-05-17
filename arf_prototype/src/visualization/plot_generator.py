"""Matplotlib/seaborn tabanlı grafik üretimi.

Her grafik tek bir PNG → run klasörü altına veya `results/comparisons` altına.
Dashboard `gr.Plot` ile figure objelerini kullanır — PNG dosyaları da yazılır.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..utils import ensure_dir

sns.set_theme(style="whitegrid", context="paper")
PALETTE = {
    "baseline":  "#6b7280",
    "A1_PCR":    "#3b82f6",
    "A2_LI_SCR": "#1d4ed8",
    "A3_RBR":    "#0ea5e9",
    "B1_MI_S3T": "#22c55e",
    "B2_CMA":    "#15803d",
    "B3_FWML":   "#84cc16",
    "C1_PCUN":   "#f97316",
    "C2_EP_T":   "#ea580c",
    "C3_CHA":    "#dc2626",
}


def _color(method: str) -> str:
    return PALETTE.get(method, "#888888")


class PlotGenerator:
    def __init__(self, save_dir: str | Path):
        self.save_dir = Path(ensure_dir(save_dir))

    def _save(self, fig, filename: str) -> str:
        path = self.save_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    # ─────────────────────────────────────────────────────
    def loss_curve(self, epoch_df: pd.DataFrame, run_id: str) -> str:
        fig, ax = plt.subplots(figsize=(9, 5))
        if "train_loss" in epoch_df.columns:
            ax.plot(epoch_df["epoch"], epoch_df["train_loss"], "-o", label="Train", color="#3b82f6")
        if "val_loss" in epoch_df.columns:
            ax.plot(epoch_df["epoch"], epoch_df["val_loss"], "-s", label="Val", color="#ef4444")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Loss curve — {run_id}")
        ax.legend()
        return self._save(fig, f"loss_curve_{run_id}.png")

    def ppl_curve(self, epoch_df: pd.DataFrame, run_id: str) -> str:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epoch_df["epoch"], epoch_df["val_ppl"], "-o", color="#1d4ed8")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation PPL")
        ax.set_title(f"Validation perplexity — {run_id}")
        return self._save(fig, f"ppl_curve_{run_id}.png")

    def ppl_comparison(self, summary_df: pd.DataFrame, dataset: str = "") -> str:
        """summary_df: ['method','ppl_mean','ppl_std',...]"""
        fig, ax = plt.subplots(figsize=(10, 6))
        df = summary_df.sort_values("ppl_mean")
        colors = [_color(m) for m in df["method"]]
        ax.bar(
            df["method"], df["ppl_mean"],
            yerr=df.get("ppl_std", 0), capsize=4, color=colors, edgecolor="black"
        )
        # Baseline çizgisi
        if "baseline" in df["method"].values:
            base = df.loc[df["method"] == "baseline", "ppl_mean"].iloc[0]
            ax.axhline(base, color="red", linestyle="--", alpha=0.6, label="baseline")
            ax.legend()
        ax.set_ylabel("Test perplexity (mean ± std)")
        ax.set_title(f"Method comparison{(' — ' + dataset) if dataset else ''}")
        plt.xticks(rotation=30, ha="right")
        return self._save(fig, f"ppl_compare_{dataset or 'all'}.png")

    def seed_variance_box(self, method_runs: Dict[str, List[float]]) -> str:
        fig, ax = plt.subplots(figsize=(10, 6))
        labels, data = list(method_runs.keys()), list(method_runs.values())
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, m in zip(bp["boxes"], labels):
            patch.set_facecolor(_color(m))
            patch.set_alpha(0.6)
        ax.set_ylabel("Test PPL")
        ax.set_title("Seed variance per method (5-seed runs)")
        plt.xticks(rotation=30, ha="right")
        return self._save(fig, "seed_variance.png")

    def speed_accuracy(self, summary_df: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(9, 6))
        for _, row in summary_df.iterrows():
            ax.scatter(
                row["duration_mean"],
                row["ppl_mean"],
                color=_color(row["method"]),
                s=120,
                edgecolors="black",
                label=row["method"],
            )
            ax.annotate(row["method"], (row["duration_mean"], row["ppl_mean"]), fontsize=8)
        ax.set_xlabel("Training time (s)")
        ax.set_ylabel("Test PPL")
        ax.set_title("Speed–accuracy trade-off")
        return self._save(fig, "speed_accuracy.png")

    def param_count(self, summary_df: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(10, 5))
        df = summary_df.sort_values("params_mean")
        colors = [_color(m) for m in df["method"]]
        ax.bar(df["method"], df["params_mean"] / 1e6, color=colors, edgecolor="black")
        ax.set_ylabel("Total parameters (M)")
        ax.set_title("Parameter count per method")
        plt.xticks(rotation=30, ha="right")
        return self._save(fig, "param_count.png")

    def domain_shift_heatmap(self, matrix: pd.DataFrame) -> str:
        """matrix: rows=methods, columns=domains, values=PPL."""
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=ax)
        ax.set_title("Domain shift — PPL matrix")
        return self._save(fig, "domain_shift.png")

    def stability(self, step_df: pd.DataFrame, run_id: str, window: int = 20) -> str:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(step_df["step"], step_df["train_loss"], alpha=0.4, label="raw")
        if len(step_df) > window:
            rolling = step_df["train_loss"].rolling(window).mean()
            ax.plot(step_df["step"], rolling, color="red", label=f"{window}-step MA")
        ax.set_xlabel("Step")
        ax.set_ylabel("Train loss")
        ax.set_title(f"Training stability — {run_id}")
        ax.legend()
        return self._save(fig, f"stability_{run_id}.png")
