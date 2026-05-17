"""Results sekmesi — DataFrame'le filtreli liste."""

from __future__ import annotations

import os

import gradio as gr
import pandas as pd

from src.logging_utils.result_store import ResultStore


def _store():
    drive = os.environ.get("DRIVE_BASE", "/content/drive/MyDrive/arf_results")
    return ResultStore(drive)


def _list_runs(status, method):
    store = _store()
    filters = {}
    if status and status != "all":
        filters["status"] = status
    if method and method != "all":
        filters["method"] = method
    df = store.runs_df(**filters)
    if df.empty:
        return df
    cols = [
        "run_id", "method", "seed", "status", "test_ppl", "test_bpc",
        "duration_seconds", "train_dataset", "eval_dataset",
        "total_params", "domain_shift_delta", "started_at",
    ]
    keep = [c for c in cols if c in df.columns]
    return df[keep]


def _run_details(run_id):
    if not run_id:
        return "Run ID gir.", None
    store = _store()
    info = store.get_run(run_id)
    if not info:
        return f"Bulunamadı: {run_id}", None
    epoch_df = store.epoch_metrics_df(run_id)
    text = "\n".join([f"{k}: {v}" for k, v in info.items()])
    return text, epoch_df


def _delete(run_id):
    if not run_id:
        return "Run ID gir."
    _store().delete_run(run_id)
    return f"🗑 Silindi: {run_id}"


def build_results_panel():
    from src.config import list_methods

    with gr.Column():
        gr.Markdown("### 📊 Tüm Run'lar")
        with gr.Row():
            status_dd = gr.Dropdown(
                ["all", "running", "completed", "failed"],
                value="all", label="Status",
            )
            method_dd = gr.Dropdown(
                ["all"] + list_methods(), value="all", label="Method",
            )
            refresh_btn = gr.Button("🔄 Listele")
        table = gr.Dataframe(interactive=False, wrap=True)

        gr.Markdown("### 🔍 Run Detayı")
        with gr.Row():
            run_id_box = gr.Textbox(label="Run ID")
            detail_btn = gr.Button("Detay göster")
            delete_btn = gr.Button("🗑 Sil", variant="stop")
        details_text = gr.Textbox(label="Run özet", lines=15, interactive=False)
        epoch_table = gr.Dataframe(label="Epoch metrikleri", interactive=False)

        refresh_btn.click(_list_runs, [status_dd, method_dd], [table])
        detail_btn.click(_run_details, [run_id_box], [details_text, epoch_table])
        delete_btn.click(_delete, [run_id_box], [details_text])
