"""Gradio dashboard giriş noktası — Colab notebook'unda çağrılır.

Kullanım:
    from dashboard.app import create_app
    app = create_app()
    app.launch(share=True)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Repo root'u path'e ekle
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr

from dashboard.components.run_panel import build_run_panel
from dashboard.components.batch_panel import build_batch_panel
from dashboard.components.results_panel import build_results_panel
from dashboard.components.compare_panel import build_compare_panel
from dashboard.components.plots_panel import build_plots_panel
from dashboard.components.config_panel import build_config_panel


CUSTOM_CSS = """
.gradio-container { max-width: 1400px !important; }
"""


def create_app() -> gr.Blocks:
    with gr.Blocks(title="ARF Prototype — Dashboard", css=CUSTOM_CSS) as app:
        gr.Markdown(
            "# 🧪 Attention Replacement Framework — Prototype\n"
            "9 aday yöntem + baseline · Colab + Drive entegrasyonu"
        )
        with gr.Tabs():
            with gr.Tab("🚀 Run"):
                build_run_panel()
            with gr.Tab("📋 Batch"):
                build_batch_panel()
            with gr.Tab("📊 Results"):
                build_results_panel()
            with gr.Tab("📈 Compare"):
                build_compare_panel()
            with gr.Tab("🗂 Plots"):
                build_plots_panel()
            with gr.Tab("⚙ Config"):
                build_config_panel()

        gr.Markdown(
            "---\n"
            "**ARF Prototype v0.1** · Drive base: "
            f"`{os.environ.get('DRIVE_BASE', '/content/drive/MyDrive/arf_results')}`"
        )
    return app


if __name__ == "__main__":
    create_app().launch(server_name="0.0.0.0", server_port=7860, share=True)
