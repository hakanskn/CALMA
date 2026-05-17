"""Notebook üretici — 5 .ipynb dosyasını programatik üret.

Çalıştır: `python scripts/build_notebooks.py`
Çıkış: notebooks/00_setup.ipynb ... 04_dashboard.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.rstrip().split("\n") if "\n" in src else [src.rstrip()],
    }


def md_cell(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.split("\n"),
    }


def make_notebook(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "machine_shape": "hm"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(name: str, cells: list) -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOKS_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=2, ensure_ascii=False)
    print(f"✓ {path}")


# ─────────────────────────────────────────────────────────
# 00_setup
# ─────────────────────────────────────────────────────────
setup_cells = [
    md_cell(
        "# 00 — Setup\n"
        "Her Colab session'ının başında **çalıştır**.\n"
        "1. Drive mount\n"
        "2. GitHub'dan repo pull\n"
        "3. requirements.txt install\n"
        "4. Drive dizinleri oluştur\n"
        "5. .env dosyası yaz\n"
        "6. GPU bilgisi"
    ),
    code_cell(
        "# 1) Google Drive mount\n"
        "from google.colab import drive\n"
        "drive.mount('/content/drive', force_remount=False)"
    ),
    code_cell(
        "# 2) Repo klonla veya güncelle\n"
        "import os\n"
        "REPO_URL = 'https://github.com/hakanskn/CALMA.git'\n"
        "REPO_DIR = '/content/arf_prototype'\n"
        "\n"
        "if not os.path.exists(REPO_DIR):\n"
        "    # CALMA repo'su içinde arf_prototype/ alt klasörü var\n"
        "    !git clone {REPO_URL} /content/CALMA\n"
        "    !cp -r /content/CALMA/arf_prototype /content/arf_prototype\n"
        "else:\n"
        "    !cd /content/CALMA && git pull && cp -r arf_prototype/* /content/arf_prototype/"
    ),
    code_cell(
        "# 3) Requirements\n"
        "!pip install -q -r /content/arf_prototype/requirements.txt"
    ),
    code_cell(
        "# 4) Drive dizinleri\n"
        "DRIVE_BASE = '/content/drive/MyDrive/arf_results'\n"
        "for sub in ['runs', 'comparisons']:\n"
        "    os.makedirs(f'{DRIVE_BASE}/{sub}', exist_ok=True)\n"
        "print('Drive base:', DRIVE_BASE)"
    ),
    code_cell(
        "# 5) .env\n"
        "env = f'DRIVE_BASE={DRIVE_BASE}\\nHF_HOME={DRIVE_BASE}/hf_cache\\n'\n"
        "with open(f'{REPO_DIR}/.env', 'w') as f:\n"
        "    f.write(env)\n"
        "os.environ['DRIVE_BASE'] = DRIVE_BASE\n"
        "os.environ['HF_HOME'] = f'{DRIVE_BASE}/hf_cache'\n"
        "print('.env yazıldı.')"
    ),
    code_cell(
        "# 6) GPU\n"
        "!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv"
    ),
    code_cell(
        "# 7) Sanity check — paketleri import et\n"
        "import sys\n"
        "sys.path.insert(0, REPO_DIR)\n"
        "from src.config import ExperimentConfig, list_methods\n"
        "print('Available methods:', list_methods())\n"
        "print('Default config:', ExperimentConfig.load('baseline').method)"
    ),
]
write("00_setup.ipynb", setup_cells)


# ─────────────────────────────────────────────────────────
# 01_single_run
# ─────────────────────────────────────────────────────────
single_cells = [
    md_cell(
        "# 01 — Single Run\n"
        "Tek yöntem × tek seed manuel deney.\n"
        "Dashboard yerine notebook'tan çalıştırmak için."
    ),
    code_cell(
        "import os, sys\n"
        "sys.path.insert(0, '/content/arf_prototype')\n"
        "os.environ.setdefault('DRIVE_BASE', '/content/drive/MyDrive/arf_results')"
    ),
    code_cell(
        "from src.config import ExperimentConfig\n"
        "from src.training.trainer import run_experiment\n"
        "\n"
        "cfg = ExperimentConfig.load(\n"
        "    method='baseline',\n"
        "    overrides={\n"
        "        'seed': 42,\n"
        "        'train_dataset': 'wikitext2',\n"
        "        'eval_dataset':  'wikitext2',\n"
        "        'num_epochs': 1,\n"
        "        'batch_size': 8,\n"
        "        'limit_train_batches': 50,   # smoke test\n"
        "        'limit_eval_batches':  20,\n"
        "    },\n"
        ")\n"
        "print(cfg.to_json())"
    ),
    code_cell(
        "result = run_experiment(cfg, progress_callback=lambda ev: print(ev))\n"
        "print('\\nFinal test PPL:', result['final_metrics']['test_ppl'])"
    ),
]
write("01_single_run.ipynb", single_cells)


# ─────────────────────────────────────────────────────────
# 02_batch_run
# ─────────────────────────────────────────────────────────
batch_cells = [
    md_cell(
        "# 02 — Batch Run\n"
        "Yön grubu × seed kombinasyonlarını sırayla çalıştır."
    ),
    code_cell(
        "import os, sys\n"
        "sys.path.insert(0, '/content/arf_prototype')\n"
        "os.environ.setdefault('DRIVE_BASE', '/content/drive/MyDrive/arf_results')"
    ),
    code_cell(
        "from src.config import ExperimentConfig, DIRECTION_GROUPS\n"
        "from src.training.trainer import run_experiment\n"
        "import time, json\n"
        "\n"
        "direction = 'A'              # 'A' | 'B' | 'C' | 'ALL'\n"
        "seeds = [42, 123]            # 1–5 seed\n"
        "epochs = 1                    # ilk test için 1\n"
        "limit_batches = 100           # tam set yerine smoke test\n"
        "\n"
        "methods = DIRECTION_GROUPS[direction]\n"
        "print(f'{len(methods)} method × {len(seeds)} seed = {len(methods)*len(seeds)} run')"
    ),
    code_cell(
        "results = []\n"
        "for m in methods:\n"
        "    for s in seeds:\n"
        "        cfg = ExperimentConfig.load(method=m, overrides={\n"
        "            'seed': s, 'num_epochs': epochs,\n"
        "            'limit_train_batches': limit_batches,\n"
        "            'limit_eval_batches': limit_batches // 2,\n"
        "            'run_id': f'run_{m}_s{s}_' + time.strftime('%Y%m%d_%H%M%S'),\n"
        "        })\n"
        "        print(f'▶ {m} seed={s}')\n"
        "        try:\n"
        "            r = run_experiment(cfg)\n"
        "            results.append((m, s, r['final_metrics']['test_ppl']))\n"
        "        except Exception as e:\n"
        "            print(f'  ✗ FAILED: {e}')\n"
        "            results.append((m, s, None))\n"
        "\n"
        "print('\\nSummary:')\n"
        "for m, s, ppl in results:\n"
        "    print(f'  {m} seed={s}: PPL={ppl}')"
    ),
]
write("02_batch_run.ipynb", batch_cells)


# ─────────────────────────────────────────────────────────
# 03_analysis
# ─────────────────────────────────────────────────────────
analysis_cells = [
    md_cell(
        "# 03 — Analysis\n"
        "Tamamlanan run'lardan karşılaştırma tabloları + grafikleri üret."
    ),
    code_cell(
        "import os, sys\n"
        "sys.path.insert(0, '/content/arf_prototype')\n"
        "os.environ.setdefault('DRIVE_BASE', '/content/drive/MyDrive/arf_results')\n"
        "from src.logging_utils.result_store import ResultStore\n"
        "from src.visualization.plot_generator import PlotGenerator\n"
        "from pathlib import Path\n"
        "\n"
        "store = ResultStore(os.environ['DRIVE_BASE'])\n"
        "runs = store.runs_df(status='completed')\n"
        "runs"
    ),
    code_cell(
        "# Tüm yöntemleri karşılaştır\n"
        "methods = list(runs['method'].unique()) if not runs.empty else []\n"
        "summary = store.compare_methods(methods)\n"
        "summary"
    ),
    code_cell(
        "# Grafik üret\n"
        "plotter = PlotGenerator(Path(os.environ['DRIVE_BASE']) / 'comparisons')\n"
        "if not summary.empty:\n"
        "    p1 = plotter.ppl_comparison(summary, dataset='wikitext2')\n"
        "    p2 = plotter.speed_accuracy(summary)\n"
        "    p3 = plotter.param_count(summary)\n"
        "    print('Saved:', p1, p2, p3)"
    ),
    code_cell(
        "from IPython.display import Image\n"
        "Image(p1)"
    ),
]
write("03_analysis.ipynb", analysis_cells)


# ─────────────────────────────────────────────────────────
# 04_dashboard
# ─────────────────────────────────────────────────────────
dashboard_cells = [
    md_cell(
        "# 04 — Dashboard\n"
        "Gradio panelini başlatır. `share=True` ile public ngrok URL üretir.\n"
        "URL'yi tarayıcıda aç → tüm run'ları buradan yönet."
    ),
    code_cell(
        "import os, sys\n"
        "sys.path.insert(0, '/content/arf_prototype')\n"
        "os.environ.setdefault('DRIVE_BASE', '/content/drive/MyDrive/arf_results')"
    ),
    code_cell(
        "from dashboard.app import create_app\n"
        "app = create_app()\n"
        "app.launch(\n"
        "    share=True,\n"
        "    debug=False,\n"
        "    server_name='0.0.0.0',\n"
        "    server_port=7860,\n"
        ")"
    ),
]
write("04_dashboard.ipynb", dashboard_cells)

print("\n✅ 5 notebook üretildi.")
