"""Plots sekmesi — bir run'ın grafiklerini üret + listele."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from src.logging_utils.result_store import ResultStore
from src.visualization.plot_generator import PlotGenerator


def _store():
    drive = os.environ.get("DRIVE_BASE", "/content/drive/MyDrive/arf_results")
    return ResultStore(drive)


def _generate(run_id):
    if not run_id:
        return [], "Run ID gir."
    drive = os.environ.get("DRIVE_BASE", "/content/drive/MyDrive/arf_results")
    plots_dir = Path(drive) / "runs" / run_id / "plots"
    plotter = PlotGenerator(plots_dir)
    store = _store()

    epoch_df = store.epoch_metrics_df(run_id)
    step_df = store.step_metrics_df(run_id)

    paths = []
    if not epoch_df.empty:
        paths.append(plotter.loss_curve(epoch_df, run_id))
        paths.append(plotter.ppl_curve(epoch_df, run_id))
    if not step_df.empty:
        paths.append(plotter.stability(step_df, run_id))

    if not paths:
        return [], "Bu run için metrik bulunamadı."
    return paths, f"✅ {len(paths)} grafik üretildi → {plots_dir}"


def _list_existing(run_id):
    if not run_id:
        return []
    drive = os.environ.get("DRIVE_BASE", "/content/drive/MyDrive/arf_results")
    plots_dir = Path(drive) / "runs" / run_id / "plots"
    if not plots_dir.exists():
        return []
    return [str(p) for p in sorted(plots_dir.glob("*.png"))]


def build_plots_panel():
    with gr.Column():
        gr.Markdown("### 🗂 Run Grafikleri")
        run_id_box = gr.Textbox(label="Run ID")
        with gr.Row():
            generate_btn = gr.Button("📈 Grafik Üret", variant="primary")
            list_btn = gr.Button("🔍 Mevcutları Listele")
        msg = gr.Textbox(label="Mesaj", interactive=False)
        gallery = gr.Gallery(label="Grafikler", columns=2, height="auto")

        generate_btn.click(_generate, [run_id_box], [gallery, msg])
        list_btn.click(_list_existing, [run_id_box], [gallery])
