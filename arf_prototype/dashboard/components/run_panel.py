"""Run sekmesi — tek deney başlatma + canlı log + ETA + son metrikler."""

from __future__ import annotations

import gradio as gr
import yaml

from src.config import (
    CONFIGS_DIR,
    METHOD_REGISTRY,
    ExperimentConfig,
    list_datasets,
    list_methods,
)
from dashboard.state import state


def _load_method_defaults(method: str) -> dict:
    path = CONFIGS_DIR / "methods" / f"{_method_file(method)}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _method_file(method: str) -> str:
    if method == "baseline":
        return "baseline"
    prefix, rest = method.split("_", 1)
    return f"{prefix}_{rest.lower()}"


def _start_single(method, train_ds, eval_ds, seed, lr, batch_size, epochs, max_seq_len, method_params_yaml, limit_train, limit_eval):
    overrides = {
        "seed": int(seed),
        "learning_rate": float(lr),
        "batch_size": int(batch_size),
        "num_epochs": int(epochs),
        "max_seq_len": int(max_seq_len),
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
    }
    if limit_train and int(limit_train) > 0:
        overrides["limit_train_batches"] = int(limit_train)
    if limit_eval and int(limit_eval) > 0:
        overrides["limit_eval_batches"] = int(limit_eval)
    if method_params_yaml.strip():
        try:
            extra = yaml.safe_load(method_params_yaml)
            if isinstance(extra, dict):
                overrides["method_params"] = extra
        except Exception as e:
            return f"❌ method_params YAML parse hatası: {e}", "{}"

    cfg = ExperimentConfig.load(method=method, overrides=overrides)
    run_id = state.submit_single(cfg)
    return f"✅ Kuyruğa eklendi: {run_id}\n{cfg.to_json()}", run_id


def _refresh_status():
    snap = state.snapshot()
    if not snap["current"]:
        return "⏸ Aktif run yok.", "", ""
    c = snap["current"]
    log = c["log_tail"]
    last = c["last_event"]
    summary = (
        f"🏃 {c['method']} (seed={c['seed']}) — status={c['status']}\n"
        f"run_id={c['run_id']}\n"
        f"queue={snap['queue_size']}"
    )
    if last:
        summary += f"\nLast event: {last.get('event')}"
        for k in ("step", "epoch", "train_loss", "val_ppl", "val_loss"):
            if k in last:
                summary += f"  {k}={last[k]}"
    return summary, log, c["run_id"]


def _stop():
    state.stop()
    return "⏹ Stop sinyali gönderildi (mevcut run bittiğinde duracak)."


def build_run_panel():
    with gr.Column():
        gr.Markdown("### 🚀 Tek Run Başlat")
        with gr.Row():
            method_dd = gr.Dropdown(
                choices=list_methods(), value="baseline", label="Yöntem"
            )
            method_info = gr.Markdown("")
        with gr.Row():
            train_ds = gr.Dropdown(choices=list_datasets(), value="wikitext2", label="Train Dataset")
            eval_ds = gr.Dropdown(choices=list_datasets(), value="wikitext2", label="Eval Dataset")
        with gr.Row():
            seed = gr.Number(value=42, label="Seed", precision=0)
            lr = gr.Number(value=5e-5, label="Learning Rate")
            batch_size = gr.Slider(4, 64, value=16, step=2, label="Batch Size")
            epochs = gr.Slider(1, 10, value=3, step=1, label="Epoch")
            max_seq_len = gr.Slider(64, 1024, value=512, step=64, label="Max Seq Len")
        with gr.Row():
            limit_train = gr.Number(value=0, label="Limit Train Batches (0=tam)", precision=0)
            limit_eval = gr.Number(value=0, label="Limit Eval Batches (0=tam)", precision=0)
        method_params = gr.Code(
            value="",
            language="yaml",
            label="method_params (YAML override — boş bırakılırsa default)",
            lines=8,
        )
        with gr.Row():
            start_btn = gr.Button("🚀 Run Başlat", variant="primary")
            refresh_btn = gr.Button("🔄 Durumu Yenile")
            stop_btn = gr.Button("⏹ Durdur", variant="stop")
        run_id_box = gr.Textbox(label="Aktif/Son Run ID", interactive=False)
        status_box = gr.Textbox(label="Durum", lines=4, interactive=False)
        log_box = gr.Textbox(label="Canlı Log (son 25 satır)", lines=20, interactive=False)

        def _on_method_change(m):
            cfg = _load_method_defaults(m)
            text = f"**{m}** — {METHOD_REGISTRY.get(m, '')}\n\n"
            text += f"Açıklama: {cfg.get('description', '')}\n"
            text += f"Beklenen ek parametre: {cfg.get('expected_extra_params', 0):,}"
            params_default = yaml.safe_dump(cfg.get("method_params", {}), allow_unicode=True, sort_keys=False)
            return text, params_default

        method_dd.change(_on_method_change, inputs=[method_dd], outputs=[method_info, method_params])

        start_btn.click(
            _start_single,
            inputs=[method_dd, train_ds, eval_ds, seed, lr, batch_size, epochs,
                    max_seq_len, method_params, limit_train, limit_eval],
            outputs=[status_box, run_id_box],
        )
        refresh_btn.click(_refresh_status, inputs=[], outputs=[status_box, log_box, run_id_box])
        stop_btn.click(_stop, inputs=[], outputs=[status_box])

    return method_dd
