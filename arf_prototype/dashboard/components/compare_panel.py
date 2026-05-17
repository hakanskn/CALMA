"""Compare sekmesi — 2+ yöntemi seç → karşılaştırma grafikleri."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from src.logging_utils.result_store import ResultStore
from src.visualization.plot_generator import PlotGenerator


def _store():
    drive = os.environ.get("DRIVE_BASE", "/content/drive/MyDrive/arf_results")
    return ResultStore(drive)


def _compare(methods, dataset_filter):
    if not methods:
        return None, None, None, "Yöntem seç."
    store = _store()
    df = store.compare_methods(methods)
    if df.empty:
        return None, None, None, "Tamamlanan run yok."

    drive = os.environ.get("DRIVE_BASE", "/content/drive/MyDrive/arf_results")
    save_dir = Path(drive) / "comparisons"
    plotter = PlotGenerator(save_dir)
    ppl_path = plotter.ppl_comparison(df, dataset=dataset_filter or "")
    sa_path = plotter.speed_accuracy(df)
    pc_path = plotter.param_count(df)

    msg = f"{len(methods)} yöntem, {df['n_runs'].sum()} run karşılaştırıldı."
    return ppl_path, sa_path, pc_path, msg


def _summary_table(methods):
    if not methods:
        return pd.DataFrame()
    return _store().compare_methods(methods)


def build_compare_panel():
    from src.config import list_methods

    with gr.Column():
        gr.Markdown("### 📈 Yöntem Karşılaştırma")
        methods = gr.CheckboxGroup(choices=list_methods(), value=["baseline"], label="Yöntemler")
        dataset_filter = gr.Textbox(label="Dataset etiketi (grafiklerde gösterim)", value="")
        run_btn = gr.Button("📈 Karşılaştır", variant="primary")
        msg = gr.Textbox(label="Mesaj", interactive=False)
        with gr.Row():
            ppl_img = gr.Image(label="PPL Comparison", type="filepath")
            sa_img = gr.Image(label="Speed–Accuracy", type="filepath")
        pc_img = gr.Image(label="Parameter Count", type="filepath")
        table = gr.Dataframe(label="Özet tablo (mean ± std)", interactive=False)

        run_btn.click(_compare, [methods, dataset_filter], [ppl_img, sa_img, pc_img, msg])
        run_btn.click(_summary_table, [methods], [table])
