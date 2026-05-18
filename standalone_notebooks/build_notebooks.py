"""Standalone notebook üreticisi.

Çalıştır:
    python build_notebooks.py

Çıkış:
    standalone_notebooks/00_baseline.ipynb
    standalone_notebooks/01_A1_PCR.ipynb
    ...
    standalone_notebooks/09_C3_CHA.ipynb
    standalone_notebooks/10_compare_all.ipynb

Her notebook 7 hücre, ortak şablon, sadece method-specific kısımları değişir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────
# Cell helpers
# ─────────────────────────────────────────────────────────
def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split(src),
    }


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _split(src)}


def _split(src: str) -> List[str]:
    lines = src.rstrip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else [""]


def notebook(cells: List[dict]) -> dict:
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


def write_notebook(filename: str, cells: List[dict]) -> None:
    path = HERE / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook(cells), f, indent=2, ensure_ascii=False)
    print(f"  wrote {filename}  ({len(cells)} cells)")


# ─────────────────────────────────────────────────────────
# Ortak hücreler
# ─────────────────────────────────────────────────────────
SETUP_CELL = """\
# ─── HÜCRE 1: Setup ────────────────────────────────────────────────────
# Colab tespit + pip install + Drive mount (opsiyonel) + arf_lib import
import os, sys, importlib
IS_COLAB = "google.colab" in sys.modules

if IS_COLAB:
    # Colab'da arf_lib.py'nin yerini bul — varsayım:
    # repo /content/CALMA/standalone_notebooks/ altında klonlu
    REPO_BASE = "/content/CALMA"
    NB_DIR = f"{REPO_BASE}/standalone_notebooks"
    if not os.path.exists(NB_DIR):
        !git clone https://github.com/hakanskn/CALMA.git {REPO_BASE} 2>&1 | tail -5
    sys.path.insert(0, NB_DIR)

    # Drive mount (sonuçları Drive'a yedeklemek istersen PARAMS'ta results_root değiştir)
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
    except Exception as e:
        print("Drive mount skipped:", e)

    !pip install -q transformers==4.41.0 datasets==2.19.0 tokenizers==0.19.0 accelerate==0.30.0 matplotlib seaborn
else:
    # Lokal: arf_lib.py notebook ile aynı klasörde
    NB_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    sys.path.insert(0, NB_DIR)

import arf_lib
importlib.reload(arf_lib)
print(f"arf_lib loaded from: {arf_lib.__file__}")
print(f"GPU info: {arf_lib.gpu_info()}")"""


IMPORTS_CELL = """\
# ─── HÜCRE 3: Imports ──────────────────────────────────────────────────
import math, time, json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arf_lib import (
    seed_everything, get_device, gpu_info, gpu_peak_mb,
    GPT2Wrapper, BaseRouterAttention,
    StandaloneTrainer, load_text_dataloaders,
    run_pipeline, save_results, append_to_index, make_plots,
)

seed_everything(PARAMS["seed"])
print(f"Seed set: {PARAMS['seed']}")
print(f"Device: {get_device()}")"""


RUN_CELL = """\
# ─── HÜCRE 6: Run ──────────────────────────────────────────────────────
# Tek satır: model + opsiyonel adapter inşa et, pipeline çalıştır
model, adapter = build_model_and_adapter(PARAMS)
run_dir, result = run_pipeline(model, PARAMS, adapter=adapter)
fm = result["final_metrics"]
print("\\n" + "=" * 60)
print(f"RUN DIR: {run_dir}")
print(f"Final test PPL:             {fm['test_ppl']:.4f}")
print(f"Final test BPC (per-char):  {fm['test_bpc']:.4f}")
print(f"Final test bits-per-token:  {fm['test_bits_per_token']:.4f}")
print(f"chars/token:                {fm['chars_per_token']:.3f}")
print(f"Duration: {result['duration_seconds']:.1f}s")
print("=" * 60)"""


DISPLAY_CELL = """\
# ─── HÜCRE 7: Display plots (inline) ───────────────────────────────────
from IPython.display import Image, display
import os
plots_dir = run_dir / "plots"
if plots_dir.exists():
    for p in sorted(plots_dir.glob("*.png")):
        print(f"📈 {p.name}")
        display(Image(filename=str(p)))
else:
    print("Plots not generated.")
print(f"\\nAll outputs in: {run_dir}")
print(f"Index file: {Path(PARAMS['results_root']) / '_index.csv'}")"""


# ─────────────────────────────────────────────────────────
# Yöntemlere göre PARAMS bloğu + Method class kodu
# ─────────────────────────────────────────────────────────
# Her giriş: title, description, params_block, method_code, build_code
METHODS: Dict[str, Dict[str, str]] = {}


def add(key: str, title: str, desc: str, params: str, method_code: str, build_code: str) -> None:
    METHODS[key] = {
        "title": title,
        "desc": desc,
        "params": params,
        "method_code": method_code,
        "build_code": build_code,
    }


# ───────────────────────── baseline ─────────────────────────
add(
    key="00_baseline",
    title="Baseline — Standart Scaled Dot-Product Attention",
    desc="Referans: GPT-2 Small'un kendi attention'ı. Diğer 9 yöntemin karşılaştırma noktası.",
    params="""\
# ─── HÜCRE 2: PARAMETRELER ─────────────────────────────────────────────
# Tüm hiperparametreler tek noktada. Sadece bu hücreyi düzenleyerek
# deneyi kontrol edebilirsin.

PARAMS = {
    # ─ Genel
    "method_name":   "baseline",            # bu notebook'a özgü, değiştirme
    "run_name":      "wt2_seed42",          # results/ altındaki klasör adı
    "seed":          42,
    "results_root":  "./results",           # LOKAL dizin (notebook yanında)

    # ─ Model
    "model_name":    "gpt2",                # gpt2 | gpt2-medium (RAM riskli)
    "tokenizer_name": "gpt2",

    # ─ Veri
    "dataset":       "wikitext2",           # wikitext2 | wikitext103 | ptb
    "max_seq_len":   512,
    "batch_size":    16,
    "num_workers":   2,

    # ─ Eğitim
    "learning_rate": 5e-5,
    "num_epochs":    3,
    "warmup_steps":  100,
    "grad_clip":     1.0,
    "weight_decay":  0.01,

    # ─ Smoke test (None = tam veri)
    "limit_train_batches": 100,             # ilk denemede küçük tut
    "limit_eval_batches":  50,

    # ─ Log / Checkpoint
    "log_every_n_steps":         25,
    "save_checkpoints":          False,
    "checkpoint_every_n_steps":  500,
    "keep_last_n_checkpoints":   3,

    # ─ Method-specific (baseline'da yok)
    # Baseline GPT-2'nin kendi attention'ını olduğu gibi kullanır.
}""",
    method_code="""\
# ─── HÜCRE 4: Method ───────────────────────────────────────────────────
# Baseline: GPT-2 Small'un standart attention'ını değiştirmiyoruz.
# Bu hücrede herhangi bir custom sınıf tanımı yok.
print("Baseline: HuggingFace GPT2 default attention kullanılacak.")""",
    build_code="""\
# ─── HÜCRE 5: Build model & adapter ────────────────────────────────────
def build_model_and_adapter(params):
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(params["model_name"])
    return model, None""",
)


# ───────────────────────── A1 PCR ─────────────────────────
add(
    key="01_A1_PCR",
    title="A1 — Predictive Coding Router (PCR)",
    desc="Routing skoru: score(i, j) = 1 / (||x_j - f_θ(x_i)||² + δ). Predictor MLP her token için bir sonraki token'ı tahmin eder, tahmin hatası ne kadar büyükse o token'a o kadar çok bakılır.",
    params="""\
# ─── HÜCRE 2: PARAMETRELER ─────────────────────────────────────────────
PARAMS = {
    "method_name":   "A1_PCR",
    "run_name":      "wt2_seed42",
    "seed":          42,
    "results_root":  "./results",

    "model_name":    "gpt2",
    "tokenizer_name": "gpt2",

    "dataset":       "wikitext2",
    "max_seq_len":   512,
    "batch_size":    16,
    "num_workers":   2,

    "learning_rate": 5e-5,
    "num_epochs":    3,
    "warmup_steps":  100,
    "grad_clip":     1.0,
    "weight_decay":  0.01,

    "limit_train_batches": 100,
    "limit_eval_batches":  50,

    "log_every_n_steps":        25,
    "save_checkpoints":         False,
    "checkpoint_every_n_steps": 500,
    "keep_last_n_checkpoints":  3,

    # ─ Method-specific (PCR)
    "pcr_predictor_dim":    256,            # predictor MLP gizli boyut
    "pcr_predictor_layers": 2,              # MLP kat sayısı
    "pcr_delta":            1e-6,           # sayısal stabilite
    "pcr_temperature":      1.0,            # softmax sıcaklığı
}""",
    method_code="""\
# ─── HÜCRE 4: Method — Predictive Coding Router ────────────────────────
class PredictiveCodingRouter(BaseRouterAttention):
    \"\"\"Score = 1 / (prediction_error + δ).
    Tek predictor MLP tüm head'lere broadcast olur (prototip).\"\"\"

    def __init__(self, gpt2_config, method_params):
        super().__init__(gpt2_config, method_params)
        mp = method_params
        pred_dim = int(mp.get("pcr_predictor_dim", 256))
        n_layers = int(mp.get("pcr_predictor_layers", 2))
        self.delta = float(mp.get("pcr_delta", 1e-6))
        self.temperature = float(mp.get("pcr_temperature", 1.0))

        layers = []
        in_d = self.embed_dim
        for _ in range(max(1, n_layers - 1)):
            layers += [nn.Linear(in_d, pred_dim), nn.ReLU()]
            in_d = pred_dim
        layers.append(nn.Linear(in_d, self.embed_dim))
        self.predictor = nn.Sequential(*layers)

    def _prediction_error(self, hidden):
        # Memory-efficient pairwise MSE:
        # ||x_j - pred_i||^2 = ||pred_i||^2 - 2 * pred_i . x_j + ||x_j||^2
        # 4D intermediate tensor olusturmaz -> O(B*S^2) bellek, O(B*S^2*H) degil.
        pred = self.predictor(hidden)                                 # [B,S,H]
        x_sq = (hidden * hidden).sum(-1)                              # [B,S_j]
        p_sq = (pred   * pred  ).sum(-1)                              # [B,S_i]
        cross = torch.matmul(pred, hidden.transpose(-1, -2))          # [B,S_i,S_j]
        errors = p_sq.unsqueeze(2) - 2.0 * cross + x_sq.unsqueeze(1)  # [B,S_i,S_j]
        return errors.clamp(min=0.0)                                  # numerik temizlik

    def _attend(self, q, k, v, hidden_states, attention_mask):
        errors = self._prediction_error(hidden_states)
        scores = 1.0 / (errors + self.delta)
        scores = scores / self.temperature
        scores = scores.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        causal = self._causal_mask(q.size(-2), k.size(-2), q.device)
        scores = scores.masked_fill(~causal, float("-inf"))
        if attention_mask is not None:
            scores = scores + attention_mask

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        out = torch.matmul(weights, v)
        self._last_stats = {"mean_err": float(errors.mean().item())}
        return out, weights

print("PredictiveCodingRouter tanımlandı.")""",
    build_code="""\
# ─── HÜCRE 5: Build model & adapter ────────────────────────────────────
def build_model_and_adapter(params):
    wrapper = GPT2Wrapper(params["model_name"])
    wrapper.inject_attention(PredictiveCodingRouter, params)
    n = sum(p.numel() for p in wrapper.parameters())
    print(f"PCR injected → toplam param: {n:,}")
    return wrapper, None""",
)


# ───────────────────────── A2 LI-SCR ─────────────────────────
add(
    key="02_A2_LI_SCR",
    title="A2 — Lateral Inhibition Sparse Router (LI-SCR)",
    desc="Wilson-Cowan iteratif inhibition + Hebbian güncelleme. Güçlü tokenlar zayıfları bastırır, sparse aktivasyon üretir.",
    params="""\
# ─── HÜCRE 2: PARAMETRELER ─────────────────────────────────────────────
PARAMS = {
    "method_name":   "A2_LI_SCR",
    "run_name":      "wt2_seed42",
    "seed":          42,
    "results_root":  "./results",

    "model_name":    "gpt2",
    "tokenizer_name": "gpt2",

    "dataset":       "wikitext2",
    "max_seq_len":   512,
    "batch_size":    16,
    "num_workers":   2,

    "learning_rate": 5e-5,
    "num_epochs":    3,
    "warmup_steps":  100,
    "grad_clip":     1.0,
    "weight_decay":  0.01,

    "limit_train_batches": 100,
    "limit_eval_batches":  50,

    "log_every_n_steps":        25,
    "save_checkpoints":         False,
    "checkpoint_every_n_steps": 500,
    "keep_last_n_checkpoints":  3,

    # ─ Method-specific (LI-SCR)
    "li_lambda":         0.5,        # inhibition şiddeti
    "li_num_iters":      3,          # W-C iterasyon
    "li_hebbian_lr":     0.01,       # inhibition matrisi update hızı
    "li_weight_decay":   0.001,      # W matrisi decay
    "li_min_activation": 0.01,       # WTA collapse önleme
}""",
    method_code="""\
# ─── HÜCRE 4: Method — Lateral Inhibition Sparse Router ────────────────
class LateralInhibitionRouter(BaseRouterAttention):
    def __init__(self, gpt2_config, method_params):
        super().__init__(gpt2_config, method_params)
        mp = method_params
        self.lam = float(mp.get("li_lambda", 0.5))
        self.num_iters = int(mp.get("li_num_iters", 3))
        self.hebbian_lr = float(mp.get("li_hebbian_lr", 0.01))
        self.weight_decay_rate = float(mp.get("li_weight_decay", 0.001))
        self.min_activation = float(mp.get("li_min_activation", 0.01))
        max_pos = gpt2_config.n_positions
        self.W_inhib = nn.Parameter(torch.randn(max_pos, max_pos) * 0.01)

    def _inhibit(self, scores, seq_len):
        W = self.W_inhib[:seq_len, :seq_len]
        alpha = scores
        for _ in range(self.num_iters):
            inhib = self.lam * torch.matmul(alpha, W.T)
            alpha = torch.relu(scores - inhib)
            alpha = alpha.clamp(min=self.min_activation)
        return alpha

    def _hebbian_update(self, alpha, seq_len):
        if not self.training:
            return
        with torch.no_grad():
            mean_alpha = alpha.mean(dim=(0, 1))
            corr = mean_alpha.T @ mean_alpha
            dW = self.hebbian_lr * corr - self.weight_decay_rate * self.W_inhib[:seq_len, :seq_len]
            self.W_inhib.data[:seq_len, :seq_len] += dW

    def _attend(self, q, k, v, hidden_states, attention_mask):
        raw = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        causal = self._causal_mask(q.size(-2), k.size(-2), q.device)
        raw = raw.masked_fill(~causal, 0.0)
        alpha = self._inhibit(raw, raw.size(-1))
        alpha = alpha.masked_fill(~causal, 0.0)
        alpha = alpha / (alpha.sum(-1, keepdim=True) + 1e-9)
        self._hebbian_update(alpha.detach(), alpha.size(-1))
        alpha_drop = self.attn_dropout(alpha)
        out = torch.matmul(alpha_drop, v)
        self._last_stats = {"sparsity": float((alpha < self.min_activation * 1.01).float().mean().item())}
        return out, alpha

print("LateralInhibitionRouter tanımlandı.")""",
    build_code="""\
def build_model_and_adapter(params):
    wrapper = GPT2Wrapper(params["model_name"])
    wrapper.inject_attention(LateralInhibitionRouter, params)
    print(f"LI-SCR injected → {sum(p.numel() for p in wrapper.parameters()):,} params")
    return wrapper, None""",
)


# ───────────────────────── A3 RBR ─────────────────────────
add(
    key="03_A3_RBR",
    title="A3 — Resonance-Based Router (RBR)",
    desc="ART tarzı memory bank + vigilance eşiği. Tokenlar prototype slot'larla rezonansa girer; eşik altı tokenlar yeni slot açar.",
    params="""\
# ─── HÜCRE 2: PARAMETRELER ─────────────────────────────────────────────
PARAMS = {
    "method_name":   "A3_RBR",
    "run_name":      "wt2_seed42",
    "seed":          42,
    "results_root":  "./results",

    "model_name":    "gpt2",
    "tokenizer_name": "gpt2",

    "dataset":       "wikitext2",
    "max_seq_len":   512,
    "batch_size":    16,
    "num_workers":   2,

    "learning_rate": 5e-5,
    "num_epochs":    3,
    "warmup_steps":  100,
    "grad_clip":     1.0,
    "weight_decay":  0.01,

    "limit_train_batches": 100,
    "limit_eval_batches":  50,

    "log_every_n_steps":        25,
    "save_checkpoints":         False,
    "checkpoint_every_n_steps": 500,
    "keep_last_n_checkpoints":  3,

    # ─ Method-specific (RBR)
    "rbr_vigilance":  0.7,        # eşik θ
    "rbr_num_slots":  64,         # bellek slot sayısı
    "rbr_slot_dim":   768,        # = hidden_size
    "rbr_beta":       1e-6,       # stabilite
    "rbr_slot_lr":    0.1,        # slot güncelleme hızı
}""",
    method_code="""\
# ─── HÜCRE 4: Method — Resonance-Based Router ──────────────────────────
class ResonanceBasedRouter(BaseRouterAttention):
    def __init__(self, gpt2_config, method_params):
        super().__init__(gpt2_config, method_params)
        mp = method_params
        self.vigilance = float(mp.get("rbr_vigilance", 0.7))
        self.num_slots = int(mp.get("rbr_num_slots", 64))
        self.slot_dim  = int(mp.get("rbr_slot_dim", self.embed_dim))
        self.beta      = float(mp.get("rbr_beta", 1e-6))
        self.slot_lr   = float(mp.get("rbr_slot_lr", 0.1))
        self.memory_bank = nn.Parameter(torch.randn(self.num_slots, self.slot_dim) * 0.02)
        self.register_buffer("slot_usage", torch.zeros(self.num_slots))
        self.value_proj = nn.Linear(self.slot_dim, self.embed_dim)

    def _resonance(self, hidden):
        m_norm = self.memory_bank / (self.memory_bank.norm(dim=-1, keepdim=True) + self.beta)
        return torch.einsum("bsh,kh->bsk", hidden, m_norm)

    def _decide(self, res):
        mask = res >= self.vigilance
        masked = res * mask.float()
        row_sum = masked.sum(-1, keepdim=True).clamp(min=1e-8)
        weights = masked / row_sum
        mismatch = mask.sum(-1) == 0
        return weights, mismatch

    def _maybe_open_slot(self, hidden, mismatch):
        if not self.training or not mismatch.any():
            return
        with torch.no_grad():
            new_proto = hidden[mismatch].mean(0)
            target = int(self.slot_usage.argmin().item())
            self.memory_bank.data[target] = (
                (1 - self.slot_lr) * self.memory_bank.data[target]
                + self.slot_lr * new_proto
            )
            self.slot_usage[target] = 0
            self.slot_usage += 1

    def _attend(self, q, k, v, hidden_states, attention_mask):
        res = self._resonance(hidden_states)
        weights, mismatch = self._decide(res)
        self._maybe_open_slot(hidden_states.detach(), mismatch)
        mem_out = torch.einsum("bsk,kh->bsh", weights, self.memory_bank)
        out = self.value_proj(mem_out)
        out = self._split_heads(out)
        self._last_stats = {"mismatch_rate": float(mismatch.float().mean().item())}
        return out, weights.unsqueeze(1)

print("ResonanceBasedRouter tanımlandı.")""",
    build_code="""\
def build_model_and_adapter(params):
    wrapper = GPT2Wrapper(params["model_name"])
    wrapper.inject_attention(ResonanceBasedRouter, params)
    print(f"RBR injected → {sum(p.numel() for p in wrapper.parameters()):,} params")
    return wrapper, None""",
)


# ───────────────────────── B1 MI-S3T ─────────────────────────
add(
    key="04_B1_MI_S3T",
    title="B1 — Meta-Initialized Self-Supervised TTT (MI-S3T)",
    desc="FOMAML çerçevesi + masked LM inner loop. Eğitimde her batch support/query'ye bölünür, outer Adam adımı orijinal ağırlık üzerinde uygulanır.",
    params="""\
# ─── HÜCRE 2: PARAMETRELER ─────────────────────────────────────────────
PARAMS = {
    "method_name":   "B1_MI_S3T",
    "run_name":      "wt2_seed42",
    "seed":          42,
    "results_root":  "./results",

    "model_name":    "gpt2",
    "tokenizer_name": "gpt2",

    "dataset":       "wikitext2",
    "max_seq_len":   512,
    "batch_size":    16,
    "num_workers":   2,

    "learning_rate": 5e-5,      # outer optimizer
    "num_epochs":    3,
    "warmup_steps":  0,         # MAML çerçevesi: scheduler yok
    "grad_clip":     1.0,
    "weight_decay":  0.01,

    "limit_train_batches": 100,
    "limit_eval_batches":  50,

    "log_every_n_steps":        25,
    "save_checkpoints":         False,
    "checkpoint_every_n_steps": 500,
    "keep_last_n_checkpoints":  3,

    # ─ Method-specific (MI-S3T)
    "mi_inner_lr":     1e-4,        # inner SGD lr
    "mi_inner_steps":  3,           # inner adım sayısı
    "mi_outer_lr":     5e-5,        # outer Adam lr
    "mi_mask_ratio":   0.15,        # masked LM oranı
    "mi_use_fomaml":   True,
}""",
    method_code="""\
# ─── HÜCRE 4: Method — Meta-Initialized Self-Supervised TTT ────────────
def _mask_tokens(input_ids, mask_token_id, mask_ratio=0.15, ignore_index=-100):
    input_ids = input_ids.clone()
    labels = input_ids.clone()
    probs = torch.full(input_ids.shape, mask_ratio, device=input_ids.device)
    mask = torch.bernoulli(probs).bool()
    labels[~mask] = ignore_index
    input_ids[mask] = mask_token_id
    return input_ids, labels


class MISSelfSupervisedTTT:
    requires_meta_loop = True
    owns_optimizer = True

    def __init__(self, model, params):
        self.model = model
        mp = params
        self.inner_lr = float(mp.get("mi_inner_lr", 1e-4))
        self.inner_steps = int(mp.get("mi_inner_steps", 3))
        self.outer_lr = float(mp.get("mi_outer_lr", 5e-5))
        self.mask_ratio = float(mp.get("mi_mask_ratio", 0.15))
        self.mask_token_id = 50256
        self.outer_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.outer_lr)

    def set_mask_token_id(self, tid):
        self.mask_token_id = tid

    def _ss_loss(self, batch):
        masked, labels = _mask_tokens(batch["input_ids"], self.mask_token_id, self.mask_ratio)
        out = self.model(input_ids=masked, attention_mask=batch.get("attention_mask"), labels=labels)
        return out.loss

    def adapt(self, context):
        was = self.model.training; self.model.train()
        inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
        for _ in range(self.inner_steps):
            loss = self._ss_loss(context)
            inner_opt.zero_grad(); loss.backward(); inner_opt.step()
        if not was: self.model.eval()

    def meta_train_step(self, support, query):
        original = {n: p.data.clone() for n, p in self.model.named_parameters()}
        self.adapt(support)
        q_loss = self._ss_loss(query)
        self.outer_optimizer.zero_grad()
        q_loss.backward()
        for n, p in self.model.named_parameters():
            p.data.copy_(original[n])
        self.outer_optimizer.step()
        return float(q_loss.item())

print("MISSelfSupervisedTTT tanımlandı.")""",
    build_code="""\
def build_model_and_adapter(params):
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(params["model_name"])
    adapter = MISSelfSupervisedTTT(model, params)
    print(f"MI-S3T adapter → {sum(p.numel() for p in model.parameters()):,} params")
    return model, adapter""",
)


# ───────────────────────── B2 CMA ─────────────────────────
add(
    key="05_B2_CMA",
    title="B2 — Contrastive Meta-Adaptation (CMA)",
    desc="NT-Xent contrastive loss + meta-training. Pozitif çift = aynı pasajın iki dropout view'ı (ESimCSE).",
    params="""\
# ─── HÜCRE 2: PARAMETRELER ─────────────────────────────────────────────
PARAMS = {
    "method_name":   "B2_CMA",
    "run_name":      "wt2_seed42",
    "seed":          42,
    "results_root":  "./results",

    "model_name":    "gpt2",
    "tokenizer_name": "gpt2",

    "dataset":       "wikitext2",
    "max_seq_len":   256,         # contrastive için kısa OK
    "batch_size":    16,
    "num_workers":   2,

    "learning_rate": 5e-5,
    "num_epochs":    3,
    "warmup_steps":  0,
    "grad_clip":     1.0,
    "weight_decay":  0.01,

    "limit_train_batches": 100,
    "limit_eval_batches":  50,

    "log_every_n_steps":        25,
    "save_checkpoints":         False,
    "checkpoint_every_n_steps": 500,
    "keep_last_n_checkpoints":  3,

    # ─ Method-specific (CMA)
    "cma_temperature":    0.07,
    "cma_inner_lr":       1e-4,
    "cma_inner_steps":    2,
    "cma_outer_lr":       5e-5,
    "cma_projection_dim": 128,
    "cma_hidden_size":    768,
}""",
    method_code="""\
# ─── HÜCRE 4: Method — Contrastive Meta-Adaptation ─────────────────────
class ContrastiveMetaAdaptation:
    requires_meta_loop = True
    owns_optimizer = True

    def __init__(self, model, params):
        self.model = model
        mp = params
        self.temperature = float(mp.get("cma_temperature", 0.07))
        self.inner_lr = float(mp.get("cma_inner_lr", 1e-4))
        self.inner_steps = int(mp.get("cma_inner_steps", 2))
        self.outer_lr = float(mp.get("cma_outer_lr", 5e-5))
        self.proj_dim = int(mp.get("cma_projection_dim", 128))
        hidden = int(mp.get("cma_hidden_size", 768))
        device = next(model.parameters()).device
        self.projector = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.proj_dim),
        ).to(device)
        self.outer_optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.projector.parameters()),
            lr=self.outer_lr,
        )

    def set_mask_token_id(self, tid):
        pass

    def _views(self, batch):
        self.model.train()
        out1 = self.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"), output_hidden_states=True)
        out2 = self.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"), output_hidden_states=True)
        h1 = out1.hidden_states[-1].mean(1)
        h2 = out2.hidden_states[-1].mean(1)
        return F.normalize(self.projector(h1), dim=-1), F.normalize(self.projector(h2), dim=-1)

    def _nt_xent(self, z1, z2):
        bs = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = (z @ z.T) / self.temperature
        mask = torch.eye(2*bs, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, float("-inf"))
        labels = torch.cat([torch.arange(bs, device=z.device)+bs, torch.arange(bs, device=z.device)])
        return F.cross_entropy(sim, labels)

    def adapt(self, context):
        was = self.model.training
        inner_opt = torch.optim.SGD(
            list(self.model.parameters()) + list(self.projector.parameters()),
            lr=self.inner_lr,
        )
        for _ in range(self.inner_steps):
            z1, z2 = self._views(context)
            loss = self._nt_xent(z1, z2)
            inner_opt.zero_grad(); loss.backward(); inner_opt.step()
        if not was: self.model.eval()

    def meta_train_step(self, support, query):
        orig_m = {n: p.data.clone() for n, p in self.model.named_parameters()}
        orig_p = {n: p.data.clone() for n, p in self.projector.named_parameters()}
        self.adapt(support)
        z1, z2 = self._views(query)
        q_loss = self._nt_xent(z1, z2)
        self.outer_optimizer.zero_grad()
        q_loss.backward()
        for n, p in self.model.named_parameters():
            p.data.copy_(orig_m[n])
        for n, p in self.projector.named_parameters():
            p.data.copy_(orig_p[n])
        self.outer_optimizer.step()
        return float(q_loss.item())

print("ContrastiveMetaAdaptation tanımlandı.")""",
    build_code="""\
def build_model_and_adapter(params):
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(params["model_name"])
    adapter = ContrastiveMetaAdaptation(model, params)
    print(f"CMA adapter → model {sum(p.numel() for p in model.parameters()):,} + proj {sum(p.numel() for p in adapter.projector.parameters()):,}")
    return model, adapter""",
)


# ───────────────────────── B3 FWML ─────────────────────────
add(
    key="06_B3_FWML",
    title="B3 — Fast Weight Meta-Learning (FWML)",
    desc="Ana ağırlıklar donar, her bloğun çıktısına LoRA-rank fast weight (A_t = U·V) eklenir. Hebbian outer-product ile gradient-free güncelleme.",
    params="""\
# ─── HÜCRE 2: PARAMETRELER ─────────────────────────────────────────────
PARAMS = {
    "method_name":   "B3_FWML",
    "run_name":      "wt2_seed42",
    "seed":          42,
    "results_root":  "./results",

    "model_name":    "gpt2",
    "tokenizer_name": "gpt2",

    "dataset":       "wikitext2",
    "max_seq_len":   512,
    "batch_size":    16,
    "num_workers":   2,

    # FWML: ana ağırlıklar donuk; bu LR sadece scheduler tutarlılığı için
    "learning_rate": 5e-5,
    "num_epochs":    3,
    "warmup_steps":  0,
    "grad_clip":     1.0,
    "weight_decay":  0.01,

    "limit_train_batches": 100,
    "limit_eval_batches":  50,

    "log_every_n_steps":        25,
    "save_checkpoints":         False,
    "checkpoint_every_n_steps": 500,
    "keep_last_n_checkpoints":  3,

    # ─ Method-specific (FWML)
    "fw_rank":          32,
    "fw_hebbian_lr":    0.01,
    "fw_decay":         0.99,
    "fw_freeze_slow":   True,
}""",
    method_code="""\
# ─── HÜCRE 4: Method — Fast Weight Meta-Learning ───────────────────────
class FastWeightMetaLearning:
    requires_meta_loop = False
    owns_optimizer = False

    def __init__(self, model, params):
        self.model = model
        mp = params
        self.rank = int(mp.get("fw_rank", 32))
        self.hebbian_lr = float(mp.get("fw_hebbian_lr", 0.01))
        self.decay = float(mp.get("fw_decay", 0.99))
        self.freeze_slow = bool(mp.get("fw_freeze_slow", True))

        hidden = model.config.n_embd
        n_layers = model.config.n_layer
        device = next(model.parameters()).device
        self.fast_U = [nn.Parameter(torch.zeros(hidden, self.rank, device=device), requires_grad=False) for _ in range(n_layers)]
        self.fast_V = [nn.Parameter(torch.zeros(self.rank, hidden, device=device), requires_grad=False) for _ in range(n_layers)]

        if self.freeze_slow:
            for p in self.model.parameters():
                p.requires_grad = False
            # Embeddings + LM head'i de dondurduk; mini bir trainable head bırakalım:
            for p in self.model.lm_head.parameters():
                p.requires_grad = True

        self._last_inputs = {}
        self._handles = []
        for idx, block in enumerate(self.model.transformer.h):
            self._handles.append(block.register_forward_hook(self._make_hook(idx)))

    def set_mask_token_id(self, tid):
        pass

    def _make_hook(self, idx):
        def hook(_mod, inputs, output):
            x = inputs[0]
            self._last_inputs[idx] = x.detach()
            U = self.fast_U[idx]; V = self.fast_V[idx]
            fast_out = (x @ V.T) @ U.T
            if isinstance(output, tuple):
                return (output[0] + fast_out,) + output[1:]
            return output + fast_out
        return hook

    def adapt(self, context):
        was = self.model.training
        self.model.eval()
        with torch.no_grad():
            self.model(input_ids=context["input_ids"], attention_mask=context.get("attention_mask"))
            for idx, x in self._last_inputs.items():
                mean_h = x.mean(dim=(0, 1))
                key = mean_h[:self.rank]
                self.fast_U[idx].data += self.hebbian_lr * torch.outer(mean_h, key)
                self.fast_U[idx].data *= self.decay
                self.fast_V[idx].data *= self.decay
        if was: self.model.train()

    def meta_train_step(self, support, query):
        return 0.0

print("FastWeightMetaLearning tanımlandı.")""",
    build_code="""\
def build_model_and_adapter(params):
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(params["model_name"])
    adapter = FastWeightMetaLearning(model, params)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"FWML → trainable {trainable:,} / total {total:,}")
    return model, adapter""",
)


# ───────────────────────── C1 PCUN ─────────────────────────
add(
    key="07_C1_PCUN",
    title="C1 — Predictive Coding Unified Network (PCUN)",
    desc="Aynı tahmin hatası (ε) hem routing hem yerel ağırlık güncellemesi için. Backprop yerine yerel update (prototip).",
    params="""\
# ─── HÜCRE 2: PARAMETRELER ─────────────────────────────────────────────
PARAMS = {
    "method_name":   "C1_PCUN",
    "run_name":      "wt2_seed42",
    "seed":          42,
    "results_root":  "./results",

    "model_name":    "gpt2",
    "tokenizer_name": "gpt2",

    "dataset":       "wikitext2",
    "max_seq_len":   512,
    "batch_size":    16,
    "num_workers":   2,

    "learning_rate": 5e-5,
    "num_epochs":    3,
    "warmup_steps":  100,
    "grad_clip":     1.0,
    "weight_decay":  0.01,

    "limit_train_batches": 100,
    "limit_eval_batches":  50,

    "log_every_n_steps":        25,
    "save_checkpoints":         False,
    "checkpoint_every_n_steps": 500,
    "keep_last_n_checkpoints":  3,

    # ─ Method-specific (PCUN)
    "pc_precision_init": 1.0,
    "pc_update_lr":      0.01,
    "pc_delta":          1e-6,
    "pc_use_local_only": False,    # True yapınca yerel update aktif
}""",
    method_code="""\
# ─── HÜCRE 4: Method — Predictive Coding Unified Network ───────────────
class PCUnifiedAttention(BaseRouterAttention):
    def __init__(self, gpt2_config, method_params):
        super().__init__(gpt2_config, method_params)
        mp = method_params
        self.delta = float(mp.get("pc_delta", 1e-6))
        self.update_lr = float(mp.get("pc_update_lr", 0.01))
        self.use_local_only = bool(mp.get("pc_use_local_only", False))
        self.layer_predictor = nn.Linear(self.embed_dim, self.embed_dim)
        self.log_precision = nn.Parameter(torch.tensor(0.0))

    def _layer_error(self, hidden):
        return hidden - self.layer_predictor(hidden)

    def _attend(self, q, k, v, hidden_states, attention_mask):
        eps = self._layer_error(hidden_states)
        err_norm = eps.norm(dim=-1)
        score_per_j = 1.0 / (err_norm + self.delta)
        scores = score_per_j.unsqueeze(1).expand(-1, q.size(-2), -1)
        scores = scores.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        causal = self._causal_mask(q.size(-2), k.size(-2), q.device)
        scores = scores.masked_fill(~causal, float("-inf"))
        if attention_mask is not None:
            scores = scores + attention_mask

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        out = torch.matmul(weights, v)

        if self.training and self.use_local_only:
            with torch.enable_grad():
                precision = torch.exp(self.log_precision)
                local_loss = 0.5 * precision * (eps ** 2).sum()
                params = list(self.layer_predictor.parameters()) + [self.log_precision]
                grads = torch.autograd.grad(local_loss, params, retain_graph=True, allow_unused=True)
                for p, g in zip(params, grads):
                    if g is not None:
                        p.data -= self.update_lr * g

        self._last_stats = {"err_mean": float(err_norm.mean().item())}
        return out, weights

print("PCUnifiedAttention tanımlandı.")""",
    build_code="""\
def build_model_and_adapter(params):
    wrapper = GPT2Wrapper(params["model_name"])
    wrapper.inject_attention(PCUnifiedAttention, params)
    print(f"PCUN injected → {sum(p.numel() for p in wrapper.parameters()):,} params")
    return wrapper, None""",
)


# ───────────────────────── C2 EP-T ─────────────────────────
add(
    key="08_C2_EP_T",
    title="C2 — Equilibrium Propagation Transformer (EP-T)",
    desc="Serbest (h⁻) ve kenetlenmiş (h⁺) faz farkı routing+learning sinyali. Prototip: target projection ile basitleştirilmiş clamp.",
    params="""\
# ─── HÜCRE 2: PARAMETRELER ─────────────────────────────────────────────
PARAMS = {
    "method_name":   "C2_EP_T",
    "run_name":      "wt2_seed42",
    "seed":          42,
    "results_root":  "./results",

    "model_name":    "gpt2",
    "tokenizer_name": "gpt2",

    "dataset":       "wikitext2",
    "max_seq_len":   512,
    "batch_size":    8,            # iki-faz forward → bellek yüksek
    "num_workers":   2,

    "learning_rate": 5e-5,
    "num_epochs":    3,
    "warmup_steps":  100,
    "grad_clip":     1.0,
    "weight_decay":  0.01,

    "limit_train_batches": 100,
    "limit_eval_batches":  50,

    "log_every_n_steps":        25,
    "save_checkpoints":         False,
    "checkpoint_every_n_steps": 500,
    "keep_last_n_checkpoints":  3,

    # ─ Method-specific (EP-T)
    "ep_nudge_beta":     0.05,
    "ep_free_iters":     10,
    "ep_clamped_iters":  4,
    "ep_use_simplified": True,
}""",
    method_code="""\
# ─── HÜCRE 4: Method — Equilibrium Propagation Transformer ─────────────
class EquilibriumPropAttention(BaseRouterAttention):
    def __init__(self, gpt2_config, method_params):
        super().__init__(gpt2_config, method_params)
        mp = method_params
        self.beta = float(mp.get("ep_nudge_beta", 0.05))
        self.use_simplified = bool(mp.get("ep_use_simplified", True))
        self.target_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _scores(self, q, k):
        return torch.matmul(q, k.transpose(-1, -2)) * self.scale

    def _attend(self, q, k, v, hidden_states, attention_mask):
        free_s = self._scores(q, k)
        if self.use_simplified:
            target = self.target_proj(hidden_states)
            clamped_h = hidden_states + self.beta * (target - hidden_states)
            qkv_c = self.c_attn(clamped_h)
            qc, kc, _ = qkv_c.split(self.embed_dim, dim=-1)
            qc = self._split_heads(qc); kc = self._split_heads(kc)
            clamped_s = self._scores(qc, kc)
        else:
            clamped_s = free_s

        combined = 0.5 * (free_s + clamped_s)
        causal = self._causal_mask(q.size(-2), k.size(-2), q.device)
        combined = combined.masked_fill(~causal, float("-inf"))
        if attention_mask is not None:
            combined = combined + attention_mask

        weights = F.softmax(combined, dim=-1)
        weights = self.attn_dropout(weights)
        out = torch.matmul(weights, v)
        self._last_stats = {"energy_diff": float((clamped_s - free_s).abs().mean().item())}
        return out, weights

print("EquilibriumPropAttention tanımlandı.")""",
    build_code="""\
def build_model_and_adapter(params):
    wrapper = GPT2Wrapper(params["model_name"])
    wrapper.inject_attention(EquilibriumPropAttention, params)
    print(f"EP-T injected → {sum(p.numel() for p in wrapper.parameters()):,} params")
    return wrapper, None""",
)


# ───────────────────────── C3 CHA ─────────────────────────
add(
    key="09_C3_CHA",
    title="C3 — Contrastive Hebbian Attention (CHA)",
    desc="Forward-Forward'ın attention versiyonu. h⁺/h⁻ farkı routing skoru; lokal Hebbian update key projection ağırlıklarına.",
    params="""\
# ─── HÜCRE 2: PARAMETRELER ─────────────────────────────────────────────
PARAMS = {
    "method_name":   "C3_CHA",
    "run_name":      "wt2_seed42",
    "seed":          42,
    "results_root":  "./results",

    "model_name":    "gpt2",
    "tokenizer_name": "gpt2",

    "dataset":       "wikitext2",
    "max_seq_len":   512,
    "batch_size":    16,
    "num_workers":   2,

    "learning_rate": 5e-5,
    "num_epochs":    3,
    "warmup_steps":  100,
    "grad_clip":     1.0,
    "weight_decay":  0.01,

    "limit_train_batches": 100,
    "limit_eval_batches":  50,

    "log_every_n_steps":        25,
    "save_checkpoints":         False,
    "checkpoint_every_n_steps": 500,
    "keep_last_n_checkpoints":  3,

    # ─ Method-specific (CHA)
    "cha_hebb_lr":        0.01,
    "cha_neg_ratio":      1.0,
    "cha_temperature":    1.0,
    "cha_clamp_strength": 0.1,
}""",
    method_code="""\
# ─── HÜCRE 4: Method — Contrastive Hebbian Attention ───────────────────
class ContrastiveHebbianAttention(BaseRouterAttention):
    def __init__(self, gpt2_config, method_params):
        super().__init__(gpt2_config, method_params)
        mp = method_params
        self.hebb_lr = float(mp.get("cha_hebb_lr", 0.01))
        self.neg_ratio = float(mp.get("cha_neg_ratio", 1.0))
        self.temperature = float(mp.get("cha_temperature", 1.0))
        self.clamp_strength = float(mp.get("cha_clamp_strength", 0.1))

    def _scores(self, q, k):
        return torch.matmul(q, k.transpose(-1, -2)) * self.scale

    def _attend(self, q, k, v, hidden_states, attention_mask):
        s_minus = self._scores(q, k)
        noise = torch.randn_like(hidden_states) * self.clamp_strength * hidden_states.std()
        clamped = hidden_states + noise
        qkv_p = self.c_attn(clamped)
        qp, kp, _ = qkv_p.split(self.embed_dim, dim=-1)
        qp = self._split_heads(qp); kp = self._split_heads(kp)
        s_plus = self._scores(qp, kp)

        delta = (s_plus - self.neg_ratio * s_minus) / self.temperature
        causal = self._causal_mask(q.size(-2), k.size(-2), q.device)
        delta = delta.masked_fill(~causal, float("-inf"))
        if attention_mask is not None:
            delta = delta + attention_mask

        weights = F.softmax(F.relu(delta), dim=-1).clamp(min=1e-9)
        weights = weights / weights.sum(-1, keepdim=True)
        weights = self.attn_dropout(weights)
        out = torch.matmul(weights, v)

        if self.training:
            with torch.no_grad():
                hp = clamped.mean(0)
                hm = hidden_states.mean(0)
                dW = self.hebb_lr * (hp.T @ hp - hm.T @ hm) / max(1, hp.size(0))
                kw = self.c_attn.weight.data[self.embed_dim:2*self.embed_dim]
                rows = min(kw.size(0), dW.size(0))
                cols = min(kw.size(1), dW.size(1))
                kw[:rows, :cols] += dW[:rows, :cols]
                self.c_attn.weight.data[self.embed_dim:2*self.embed_dim] = kw

        self._last_stats = {"delta_mean": float((s_plus - s_minus).abs().mean().item())}
        return out, weights

print("ContrastiveHebbianAttention tanımlandı.")""",
    build_code="""\
def build_model_and_adapter(params):
    wrapper = GPT2Wrapper(params["model_name"])
    wrapper.inject_attention(ContrastiveHebbianAttention, params)
    print(f"CHA injected → {sum(p.numel() for p in wrapper.parameters()):,} params")
    return wrapper, None""",
)


# ─────────────────────────────────────────────────────────
# Ortak şablonu method'a uygula
# ─────────────────────────────────────────────────────────
def build_method_notebook(key: str, meta: Dict[str, str]) -> List[dict]:
    return [
        md(f"# {meta['title']}\n\n{meta['desc']}\n\n"
           f"**Bağımsız çalışır.** Tüm parametreler Hücre 2'de toplanmıştır — sadece o hücreyi düzenleyerek deneyi kontrol edebilirsin. "
           f"Sonuçlar `./results/{key.split('_', 1)[1]}_*` altına kaydedilir."),
        code(SETUP_CELL),
        code(meta["params"]),
        code(IMPORTS_CELL),
        code(meta["method_code"]),
        code(meta["build_code"]),
        code(RUN_CELL),
        code(DISPLAY_CELL),
    ]


# ─────────────────────────────────────────────────────────
# 11. notebook — compare_all
# ─────────────────────────────────────────────────────────
COMPARE_CELLS = [
    md(
        "# Karşılaştırma — Tüm Yöntemler\n\n"
        "10 notebook'u çalıştırdıktan sonra bu notebook `results/_index.csv` üzerinden "
        "tüm yöntemleri karşılaştırma grafiklerine döker. Sonuçlar `results/_compare/` altına yazılır."
    ),
    code(
        "# Setup\n"
        "import os, sys\n"
        "NB_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()\n"
        "sys.path.insert(0, NB_DIR)\n"
        "import importlib, arf_lib; importlib.reload(arf_lib)\n"
        "print('arf_lib:', arf_lib.__file__)"
    ),
    code(
        "# Konfig\n"
        "RESULTS_ROOT = './results'         # 10 notebook'un yazdığı yer\n"
        "OUTPUT_DIR   = './results/_compare' # karşılaştırma çıktıları\n"
        "import os\n"
        "os.makedirs(OUTPUT_DIR, exist_ok=True)"
    ),
    code(
        "# _index.csv'yi yükle\n"
        "import pandas as pd\n"
        "from pathlib import Path\n"
        "idx_path = Path(RESULTS_ROOT) / '_index.csv'\n"
        "if not idx_path.exists():\n"
        "    raise FileNotFoundError(f'Run yok: {idx_path}. Önce diğer notebook\\'ları çalıştır.')\n"
        "df = pd.read_csv(idx_path)\n"
        "print(f'{len(df)} run yüklendi. Yöntemler: {sorted(df.method.unique())}')\n"
        "df.head()"
    ),
    code(
        "# Her yöntem için en iyi (min PPL) ve mean ± std\n"
        "summary = (df.groupby('method')\n"
        "             .agg(ppl_mean=('test_ppl','mean'),\n"
        "                  ppl_std =('test_ppl','std'),\n"
        "                  ppl_min =('test_ppl','min'),\n"
        "                  bpc_mean=('test_bpc','mean'),\n"
        "                  duration_mean=('duration_s','mean'),\n"
        "                  params_mean=('params_total','mean'),\n"
        "                  n_runs  =('test_ppl','count'))\n"
        "             .reset_index()\n"
        "             .sort_values('ppl_mean'))\n"
        "summary['ppl_std'] = summary['ppl_std'].fillna(0)\n"
        "summary.to_csv(f'{OUTPUT_DIR}/summary.csv', index=False)\n"
        "summary"
    ),
    code(
        "# Karşılaştırma grafikleri\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "\n"
        "PALETTE = {\n"
        "    'baseline':  '#6b7280',\n"
        "    'A1_PCR':    '#3b82f6',\n"
        "    'A2_LI_SCR': '#1d4ed8',\n"
        "    'A3_RBR':    '#0ea5e9',\n"
        "    'B1_MI_S3T': '#22c55e',\n"
        "    'B2_CMA':    '#15803d',\n"
        "    'B3_FWML':   '#84cc16',\n"
        "    'C1_PCUN':   '#f97316',\n"
        "    'C2_EP_T':   '#ea580c',\n"
        "    'C3_CHA':    '#dc2626',\n"
        "}\n"
        "\n"
        "# 1) PPL bar (mean ± std)\n"
        "fig, ax = plt.subplots(figsize=(11, 6))\n"
        "s = summary.copy()\n"
        "colors = [PALETTE.get(m, '#888888') for m in s['method']]\n"
        "ax.bar(s['method'], s['ppl_mean'], yerr=s['ppl_std'], capsize=4, color=colors, edgecolor='black')\n"
        "if 'baseline' in s['method'].values:\n"
        "    base = s.loc[s['method']=='baseline','ppl_mean'].iloc[0]\n"
        "    ax.axhline(base, color='red', linestyle='--', alpha=0.6, label='baseline')\n"
        "    ax.legend()\n"
        "ax.set_ylabel('Test PPL (mean ± std)')\n"
        "ax.set_title('Method comparison — Test perplexity')\n"
        "plt.xticks(rotation=30, ha='right'); plt.grid(axis='y', alpha=0.3)\n"
        "plt.savefig(f'{OUTPUT_DIR}/ppl_compare.png', dpi=150, bbox_inches='tight'); plt.show()"
    ),
    code(
        "# 2) Speed–accuracy scatter\n"
        "fig, ax = plt.subplots(figsize=(10, 6))\n"
        "for _, r in summary.iterrows():\n"
        "    ax.scatter(r['duration_mean'], r['ppl_mean'], color=PALETTE.get(r['method'],'#888'), s=140, edgecolors='black')\n"
        "    ax.annotate(r['method'], (r['duration_mean'], r['ppl_mean']), fontsize=8, xytext=(5,5), textcoords='offset points')\n"
        "ax.set_xlabel('Training time (s)'); ax.set_ylabel('Test PPL')\n"
        "ax.set_title('Speed vs. accuracy'); ax.grid(alpha=0.3)\n"
        "plt.savefig(f'{OUTPUT_DIR}/speed_accuracy.png', dpi=150, bbox_inches='tight'); plt.show()"
    ),
    code(
        "# 3) Parameter count\n"
        "fig, ax = plt.subplots(figsize=(11, 5))\n"
        "s2 = summary.sort_values('params_mean')\n"
        "colors2 = [PALETTE.get(m,'#888') for m in s2['method']]\n"
        "ax.bar(s2['method'], s2['params_mean']/1e6, color=colors2, edgecolor='black')\n"
        "ax.set_ylabel('Total parameters (M)'); ax.set_title('Parameter count by method')\n"
        "plt.xticks(rotation=30, ha='right'); plt.grid(axis='y', alpha=0.3)\n"
        "plt.savefig(f'{OUTPUT_DIR}/param_count.png', dpi=150, bbox_inches='tight'); plt.show()"
    ),
    code(
        "# 4) Seed varyansı (multi-seed run varsa)\n"
        "multi = df.groupby('method').filter(lambda x: len(x) >= 2)\n"
        "if not multi.empty:\n"
        "    fig, ax = plt.subplots(figsize=(11, 6))\n"
        "    groups = list(multi.groupby('method'))\n"
        "    data = [g['test_ppl'].values for _, g in groups]\n"
        "    labels = [m for m,_ in groups]\n"
        "    bp = ax.boxplot(data, labels=labels, patch_artist=True)\n"
        "    for patch, m in zip(bp['boxes'], labels):\n"
        "        patch.set_facecolor(PALETTE.get(m,'#888')); patch.set_alpha(0.6)\n"
        "    ax.set_ylabel('Test PPL'); ax.set_title('Seed variance per method')\n"
        "    plt.xticks(rotation=30, ha='right'); plt.grid(axis='y', alpha=0.3)\n"
        "    plt.savefig(f'{OUTPUT_DIR}/seed_variance.png', dpi=150, bbox_inches='tight'); plt.show()\n"
        "else:\n"
        "    print('Multi-seed run yok — seed varyansı atlanıyor.')"
    ),
    code(
        "# Özet\n"
        "print('Sonuç dizini:', OUTPUT_DIR)\n"
        "for f in sorted(os.listdir(OUTPUT_DIR)):\n"
        "    print('  •', f)"
    ),
]


# ─────────────────────────────────────────────────────────
# Üret
# ─────────────────────────────────────────────────────────
def main():
    print(f"Notebook'lar üretiliyor → {HERE}")
    for key, meta in METHODS.items():
        cells = build_method_notebook(key, meta)
        write_notebook(f"{key}.ipynb", cells)
    write_notebook("10_compare_all.ipynb", COMPARE_CELLS)
    print(f"\n✓ {len(METHODS) + 1} notebook üretildi.")


if __name__ == "__main__":
    main()
