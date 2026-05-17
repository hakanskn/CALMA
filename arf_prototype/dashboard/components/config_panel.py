"""Config sekmesi — base + method YAML'larını dashboard'dan düzenle."""

from __future__ import annotations

import gradio as gr
import yaml

from src.config import CONFIGS_DIR, list_methods


def _config_paths():
    paths = [str(CONFIGS_DIR / "base_config.yaml")]
    for m in sorted((CONFIGS_DIR / "methods").glob("*.yaml")):
        paths.append(str(m))
    for d in sorted((CONFIGS_DIR / "datasets").glob("*.yaml")):
        paths.append(str(d))
    return paths


def _load_file(path):
    if not path:
        return "Dosya seç."
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _save_file(path, content):
    if not path:
        return "Dosya seç."
    try:
        # YAML parse doğrula
        yaml.safe_load(content)
    except yaml.YAMLError as e:
        return f"❌ YAML parse hatası: {e}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"✅ Kaydedildi: {path}"


def build_config_panel():
    with gr.Column():
        gr.Markdown("### ⚙ Config Düzenle")
        path_dd = gr.Dropdown(choices=_config_paths(), label="Config dosyası")
        editor = gr.Code(language="yaml", label="YAML içerik")
        with gr.Row():
            load_btn = gr.Button("📂 Yükle")
            save_btn = gr.Button("💾 Kaydet", variant="primary")
            reload_paths_btn = gr.Button("🔄 Liste yenile")
        msg = gr.Textbox(label="Mesaj", interactive=False)

        load_btn.click(_load_file, [path_dd], [editor])
        save_btn.click(_save_file, [path_dd, editor], [msg])
        reload_paths_btn.click(lambda: gr.update(choices=_config_paths()), [], [path_dd])
