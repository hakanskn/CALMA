"""Batch Run sekmesi — A/B/C/ALL grubu × N seed."""

from __future__ import annotations

from typing import List

import gradio as gr

from src.config import DIRECTION_GROUPS, ExperimentConfig, list_methods, list_datasets
from dashboard.state import state


def _build_configs(methods: List[str], seeds: List[int], train_ds, eval_ds, batch_size, epochs, max_seq_len) -> List[ExperimentConfig]:
    cfgs = []
    for m in methods:
        for s in seeds:
            cfg = ExperimentConfig.load(
                method=m,
                overrides={
                    "seed": int(s),
                    "train_dataset": train_ds,
                    "eval_dataset": eval_ds,
                    "batch_size": int(batch_size),
                    "num_epochs": int(epochs),
                    "max_seq_len": int(max_seq_len),
                    "run_id": f"run_{m}_s{s}_{__import__('time').strftime('%Y%m%d_%H%M%S')}",
                },
            )
            cfgs.append(cfg)
    return cfgs


def _start_batch(direction, custom_methods, num_seeds, train_ds, eval_ds, batch_size, epochs, max_seq_len):
    if direction == "Custom":
        methods = [m.strip() for m in custom_methods.split(",") if m.strip()]
    else:
        methods = DIRECTION_GROUPS[direction]
    seeds_list = [42, 123, 456, 789, 1024][: int(num_seeds)]
    cfgs = _build_configs(methods, seeds_list, train_ds, eval_ds, batch_size, epochs, max_seq_len)
    state.enqueue(cfgs)
    return f"✅ {len(cfgs)} run kuyruğa eklendi.\nMethods: {methods}\nSeeds: {seeds_list}"


def _queue_status():
    snap = state.snapshot()
    cur = snap["current"]
    cur_line = (
        f"Aktif: {cur['method']} seed={cur['seed']} (run_id={cur['run_id']})"
        if cur else "Aktif run yok."
    )
    hist_lines = [
        f"  • {h['method']} seed={h['seed']} → {h['status']}" + (f" ({h['error']})" if h["error"] else "")
        for h in snap["history"]
    ]
    return (
        f"{cur_line}\nKuyrukta: {snap['queue_size']} run\n\n"
        f"Son run'lar:\n" + ("\n".join(hist_lines) if hist_lines else "  —")
    )


def build_batch_panel():
    with gr.Column():
        gr.Markdown("### 📋 Toplu Run — yön grubu × seed")
        with gr.Row():
            direction = gr.Dropdown(
                choices=["A", "B", "C", "ALL", "Custom"],
                value="A",
                label="Yön Grubu",
            )
            num_seeds = gr.Slider(1, 5, value=1, step=1, label="Seed Sayısı")
        custom_methods = gr.Textbox(
            label="Custom methods (virgülle, sadece 'Custom' seçilince)",
            placeholder="A1_PCR, A2_LI_SCR, baseline",
        )
        with gr.Row():
            train_ds = gr.Dropdown(choices=list_datasets(), value="wikitext2", label="Train Dataset")
            eval_ds = gr.Dropdown(choices=list_datasets(), value="wikitext2", label="Eval Dataset")
        with gr.Row():
            batch_size = gr.Slider(4, 64, value=16, step=2, label="Batch Size")
            epochs = gr.Slider(1, 10, value=3, step=1, label="Epoch")
            max_seq_len = gr.Slider(64, 1024, value=512, step=64, label="Max Seq Len")
        with gr.Row():
            start_btn = gr.Button("📋 Batch Başlat", variant="primary")
            refresh_btn = gr.Button("🔄 Kuyruğu Yenile")
            stop_btn = gr.Button("⏹ Durdur", variant="stop")
        status_box = gr.Textbox(label="Durum", lines=12, interactive=False)

        start_btn.click(
            _start_batch,
            inputs=[direction, custom_methods, num_seeds, train_ds, eval_ds,
                    batch_size, epochs, max_seq_len],
            outputs=[status_box],
        )
        refresh_btn.click(_queue_status, inputs=[], outputs=[status_box])
        stop_btn.click(lambda: (state.stop(), "Stop sinyali gönderildi.")[1], inputs=[], outputs=[status_box])
