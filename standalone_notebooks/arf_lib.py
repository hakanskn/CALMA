"""ARF standalone helper — tüm notebook'ların ortak Trainer/Evaluator/Data/Plot kodu.

Notebook'lar bu modülü `from arf_lib import ...` ile çağırır.
Method-specific (PCR, LI-SCR, FOMAML vs.) sınıflar notebook içinde tanımlanır.

İçindekiler:
  - seed_everything()
  - GPT2Wrapper                 GPT-2 LM head + attention inject yardımcısı
  - load_text_dataloaders()     HF dataset → train/val/test DataLoader
  - StandaloneTrainer           ortak eğitim döngüsü (custom forward hook opsiyonel)
  - StandaloneEvaluator         PPL/BPC/loss
  - save_results()              metrics+config+log JSON yaz
  - append_to_index()           _index.csv'ye satır ekle
  - make_plots()                3 standart grafik (loss/ppl/stability)
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, get_linear_schedule_with_warmup


# ─────────────────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────────────────
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_info() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"name": "CPU", "mem_gb": 0.0}
    return {
        "name": torch.cuda.get_device_name(0),
        "mem_gb": torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
    }


def gpu_peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


# ─────────────────────────────────────────────────────────
# Veri
# ─────────────────────────────────────────────────────────
DATASET_MAP: Dict[str, Dict[str, Any]] = {
    "wikitext2":   {"hf_id": "wikitext", "hf_config": "wikitext-2-raw-v1",   "text_column": "text"},
    "wikitext103": {"hf_id": "wikitext", "hf_config": "wikitext-103-raw-v1", "text_column": "text"},
    "ptb":         {"hf_id": "ptb_text_only", "hf_config": "penn_treebank",   "text_column": "sentence"},
}


class _BlockedTextDataset(Dataset):
    """Tüm metni concat edip max_seq_len bloklara böler."""

    def __init__(self, hf_split, tokenizer, text_column: str, max_len: int):
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        texts = [t for t in hf_split[text_column] if isinstance(t, str) and t.strip()]
        joined = "\n\n".join(texts)
        tokens = tokenizer(joined, return_tensors="pt").input_ids[0]
        n_blocks = max(1, tokens.size(0) // max_len)
        usable = n_blocks * max_len
        if tokens.size(0) < max_len:
            pad = torch.full((max_len - tokens.size(0),), tokenizer.eos_token_id, dtype=tokens.dtype)
            self.blocks = torch.cat([tokens, pad]).unsqueeze(0)
        else:
            self.blocks = tokens[:usable].view(n_blocks, max_len)

    def __len__(self):
        return self.blocks.size(0)

    def __getitem__(self, i):
        ids = self.blocks[i]
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


def load_text_dataloaders(
    dataset_name: str,
    tokenizer_name: str = "gpt2",
    max_seq_len: int = 512,
    batch_size: int = 16,
    num_workers: int = 2,
    limit_train_batches: Optional[int] = None,
    limit_eval_batches:  Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Any]:
    """Returns (train_loader, val_loader, test_loader, tokenizer)."""
    from datasets import load_dataset

    if dataset_name not in DATASET_MAP:
        raise ValueError(f"Bilinmeyen dataset '{dataset_name}'. Seçenekler: {list(DATASET_MAP)}")
    cfg = DATASET_MAP[dataset_name]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(cfg["hf_id"], cfg["hf_config"])
    # HF wikitext: train/validation/test mevcut
    splits = {}
    for k_target, k_src_options in (("train", ("train",)),
                                     ("val",   ("validation", "val", "dev")),
                                     ("test",  ("test",))):
        for kk in k_src_options:
            if kk in raw:
                splits[k_target] = raw[kk]
                break
    if "val" not in splits:
        # train'i %90/%10 böl
        n = len(splits["train"])
        cut = int(n * 0.9)
        splits["val"] = splits["train"].select(range(cut, n))
        splits["train"] = splits["train"].select(range(0, cut))
    if "test" not in splits:
        splits["test"] = splits["val"]

    def _ds(s):
        return _BlockedTextDataset(s, tokenizer, cfg["text_column"], max_seq_len)

    train_ds = _ds(splits["train"])
    val_ds = _ds(splits["val"])
    test_ds = _ds(splits["test"])

    def _maybe_limit(ds, n_batches):
        if n_batches is None or n_batches <= 0:
            return ds
        keep = min(len(ds), int(n_batches) * batch_size)
        return Subset(ds, list(range(keep)))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        _maybe_limit(train_ds, limit_train_batches),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        _maybe_limit(val_ds, limit_eval_batches),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        _maybe_limit(test_ds, limit_eval_batches),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin,
    )
    return train_loader, val_loader, test_loader, tokenizer


# ─────────────────────────────────────────────────────────
# GPT-2 wrapper — attention sınıfı inject etmek için
# ─────────────────────────────────────────────────────────
class GPT2Wrapper(nn.Module):
    """GPT-2 LM head + attention modülünü değiştirme arayüzü.

    AttentionClass HuggingFace `GPT2Attention.forward` signature'ına uymalı:
       forward(hidden_states, layer_past, attention_mask, head_mask,
               encoder_hidden_states, encoder_attention_mask,
               use_cache, output_attentions)
       -> (attn_output, present, [attn_weights])
    """

    def __init__(self, model_name: str = "gpt2"):
        super().__init__()
        self.model_name = model_name
        self.gpt2_config = GPT2Config.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def inject_attention(self, attention_cls: Callable[..., nn.Module], method_params: Dict[str, Any]) -> List[nn.Module]:
        modules = []
        for block in self.model.transformer.h:
            new_attn = attention_cls(self.gpt2_config, method_params)
            block.attn = new_attn
            modules.append(new_attn)
        return modules

    def forward(self, input_ids, attention_mask=None, labels=None, **kw):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kw)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────
# Ortak attention iskeleti — Yön A / C için temel sınıf
# Notebook'lar bunu import edip subclass yapabilir.
# ─────────────────────────────────────────────────────────
class BaseRouterAttention(nn.Module):
    """HF GPT2Attention uyumlu iskelet. Alt sınıf `_attend()` implement eder."""

    def __init__(self, gpt2_config, method_params: Dict[str, Any]):
        super().__init__()
        self.gpt2_config = gpt2_config
        self.method_params = method_params
        self.embed_dim = gpt2_config.n_embd
        self.num_heads = gpt2_config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.attn_dropout = nn.Dropout(gpt2_config.attn_pdrop)
        self.resid_dropout = nn.Dropout(gpt2_config.resid_pdrop)

        max_pos = gpt2_config.n_positions
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_pos, max_pos, dtype=torch.bool)).view(1, 1, max_pos, max_pos),
            persistent=False,
        )
        self._last_stats: Dict[str, Any] = {}

    def _split_heads(self, x):
        b, s, _ = x.size()
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        b, h, s, d = x.size()
        return x.transpose(1, 2).contiguous().view(b, s, h * d)

    def _qkv(self, hidden):
        qkv = self.c_attn(hidden)
        q, k, v = qkv.split(self.embed_dim, dim=-1)
        return self._split_heads(q), self._split_heads(k), self._split_heads(v)

    def _causal_mask(self, q_len, k_len, device):
        return self.bias[:, :, k_len - q_len : k_len, :k_len]  # type: ignore[attr-defined]

    def _attend(self, q, k, v, hidden_states, attention_mask):
        """Alt sınıf override eder. Returns (attn_output[b,h,s,d], weights[b,h,s,s])."""
        raise NotImplementedError

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                use_cache=False, output_attentions=False):
        q, k, v = self._qkv(hidden_states)
        if layer_past is not None:
            pk, pv = layer_past
            k = torch.cat([pk, k], dim=-2)
            v = torch.cat([pv, v], dim=-2)
        present = (k, v) if use_cache else None

        out, weights = self._attend(q, k, v, hidden_states, attention_mask)
        out = self._merge_heads(out)
        out = self.c_proj(out)
        out = self.resid_dropout(out)
        if output_attentions:
            return (out, present, weights)
        return (out, present)


# ─────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────
class StandaloneTrainer:
    """Tek-yöntemli minimal eğitim döngüsü.

    Args:
        model: GPT2Wrapper veya direkt GPT2LMHeadModel
        params: PARAMS dict (notebook'taki tek hücre)
        run_dir: sonuçların yazılacağı klasör (Path)
        log_fn: (line: str) -> None  — opsiyonel print/dosya yazıcı
        adapter: Yön B yöntemleri için opsiyonel adapter nesnesi
    """

    def __init__(
        self,
        model: nn.Module,
        params: Dict[str, Any],
        run_dir: Path,
        log_fn: Optional[Callable[[str], None]] = None,
        adapter: Optional[Any] = None,
    ):
        self.model = model
        self.params = params
        self.run_dir = run_dir
        self.device = get_device()
        self.log_fn = log_fn or (lambda s: print(s))
        self.adapter = adapter

        # Veri
        self.train_loader, self.val_loader, self.test_loader, self.tokenizer = load_text_dataloaders(
            dataset_name=params["dataset"],
            tokenizer_name=params.get("tokenizer_name", "gpt2"),
            max_seq_len=params["max_seq_len"],
            batch_size=params["batch_size"],
            num_workers=params.get("num_workers", 2),
            limit_train_batches=params.get("limit_train_batches"),
            limit_eval_batches=params.get("limit_eval_batches"),
        )
        # Eğer adapter'da mask_token_id ayarı gerekiyorsa
        if adapter is not None and hasattr(adapter, "set_mask_token_id"):
            adapter.set_mask_token_id(self.tokenizer.eos_token_id)

        # Optimizer
        if adapter is not None and getattr(adapter, "owns_optimizer", False):
            self.optimizer = None
        else:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                trainable,
                lr=params["learning_rate"],
                weight_decay=params.get("weight_decay", 0.01),
            )

        total_steps = max(1, len(self.train_loader) * params["num_epochs"])
        self.scheduler = (
            get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=params.get("warmup_steps", 100),
                num_training_steps=total_steps,
            )
            if self.optimizer is not None
            else None
        )

        # State
        self.history: Dict[str, List[Dict[str, Any]]] = {"epoch": [], "step": []}
        self._global_step = 0
        self._t_start = 0.0

        # Checkpoint dir
        self.ckpt_dir = run_dir / "checkpoints"
        if params.get("save_checkpoints", False):
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────
    def _save_checkpoint(self, step: int) -> None:
        if not self.params.get("save_checkpoints", False):
            return
        path = self.ckpt_dir / f"ckpt_step_{step:06d}.pt"
        torch.save(
            {
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            },
            path,
        )
        # rotate
        keep = self.params.get("keep_last_n_checkpoints", 3)
        ckpts = sorted(self.ckpt_dir.glob("ckpt_step_*.pt"))
        for old in ckpts[:-keep]:
            try:
                old.unlink()
            except OSError:
                pass

    def _maybe_grad_clip(self):
        clip = self.params.get("grad_clip", 0)
        if clip and clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

    # ─────────────────────────────────────────────────────
    def fit(self) -> Dict[str, Any]:
        self.model.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self._t_start = time.perf_counter()
        n_epochs = int(self.params["num_epochs"])
        log_every = int(self.params.get("log_every_n_steps", 50))
        ckpt_every = int(self.params.get("checkpoint_every_n_steps", 500))

        self.log_fn(f"▶ Training start | epochs={n_epochs} steps/epoch={len(self.train_loader)} device={self.device}")

        for epoch in range(n_epochs):
            t_ep = time.perf_counter()
            train_loss = self._train_one_epoch(epoch, log_every, ckpt_every)
            val_metrics = self.evaluate(self.val_loader)
            ep_rec = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": val_metrics["loss"],
                "val_ppl":  val_metrics["perplexity"],
                "val_bpc":  val_metrics["bits_per_char"],
                "epoch_time_s": time.perf_counter() - t_ep,
                "gpu_mem_mb": gpu_peak_mb(),
            }
            self.history["epoch"].append(ep_rec)
            self.log_fn(
                f"  ◆ Epoch {epoch+1}/{n_epochs} | train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_ppl={val_metrics['perplexity']:.2f} "
                f"time={ep_rec['epoch_time_s']:.1f}s mem={ep_rec['gpu_mem_mb']:.0f}MB"
            )

        # Test
        test_metrics = self.evaluate(self.test_loader)
        duration = time.perf_counter() - self._t_start

        result = {
            "duration_seconds": duration,
            "history": self.history,
            "final_metrics": {
                "test_loss": test_metrics["loss"],
                "test_ppl":  test_metrics["perplexity"],
                "test_bpc":  test_metrics["bits_per_char"],
            },
            "gpu_peak_mb": gpu_peak_mb(),
            "params_total": sum(p.numel() for p in self.model.parameters()),
            "params_trainable": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        self.log_fn(
            f"✓ Done | test_ppl={test_metrics['perplexity']:.2f} "
            f"test_bpc={test_metrics['bits_per_char']:.4f} "
            f"duration={duration:.1f}s"
        )
        return result

    # ─────────────────────────────────────────────────────
    def _train_one_epoch(self, epoch: int, log_every: int, ckpt_every: int) -> float:
        self.model.train()
        running, n_step = 0.0, 0

        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if self.adapter is not None and getattr(self.adapter, "requires_meta_loop", False):
                # FOMAML/CMA stili
                support, query = _split_support_query(batch)
                loss_val = self.adapter.meta_train_step(support, query)
            else:
                out = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
                loss = out.loss
                self.optimizer.zero_grad()
                loss.backward()
                self._maybe_grad_clip()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                loss_val = float(loss.item())

            running += loss_val
            n_step += 1
            self._global_step += 1

            if self._global_step % log_every == 0:
                eta = _eta_seconds(self._global_step, len(self.train_loader) * self.params["num_epochs"], self._t_start)
                self.history["step"].append({
                    "step": self._global_step, "epoch": epoch + 1,
                    "train_loss": loss_val, "eta_seconds": eta,
                })
                self.log_fn(
                    f"  step={self._global_step:>5d} ep={epoch+1} loss={loss_val:.4f} eta={_fmt_dur(eta)}"
                )

            if ckpt_every and self._global_step % ckpt_every == 0:
                self._save_checkpoint(self._global_step)

        return running / max(1, n_step)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss, total_tok = 0.0, 0
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
            )
            ntok = int(batch["labels"].ne(-100).sum().item())
            total_loss += float(out.loss.item()) * ntok
            total_tok += ntok
        avg = total_loss / max(1, total_tok)
        return {
            "loss": float(avg),
            "perplexity": float(math.exp(min(20.0, avg))),
            "bits_per_char": float(avg / math.log(2)),
            "num_tokens": int(total_tok),
        }


def _split_support_query(batch: Dict[str, torch.Tensor], ratio: float = 0.5):
    bs = batch["input_ids"].size(0)
    cut = max(1, int(bs * ratio))
    return ({k: v[:cut] for k, v in batch.items()},
            {k: v[cut:] for k, v in batch.items()})


def _eta_seconds(step, total, start):
    if step <= 0:
        return -1.0
    elapsed = time.perf_counter() - start
    rate = step / max(elapsed, 1e-9)
    return (total - step) / max(rate, 1e-9)


def _fmt_dur(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h:d}h{m:02d}m"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


# ─────────────────────────────────────────────────────────
# Sonuç kaydetme + index
# ─────────────────────────────────────────────────────────
def make_run_dir(results_root: str | Path, method: str, run_name: str) -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_run = run_name.replace(" ", "_").replace("/", "_")
    path = Path(results_root) / f"{method}_{safe_run}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results(run_dir: Path, params: Dict[str, Any], result: Dict[str, Any], log_lines: List[str]) -> None:
    # config.json
    (run_dir / "config.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    # metrics.json
    payload = {
        "started_at": result.get("started_at"),
        "duration_seconds": result.get("duration_seconds"),
        "final_metrics": result.get("final_metrics"),
        "history": result.get("history"),
        "gpu_peak_mb": result.get("gpu_peak_mb"),
        "params_total": result.get("params_total"),
        "params_trainable": result.get("params_trainable"),
        "device": result.get("device"),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    # log.txt
    (run_dir / "log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    # final_summary.txt
    final = result.get("final_metrics", {})
    summary = (
        f"METHOD: {params.get('method_name')}\n"
        f"RUN_NAME: {params.get('run_name')}\n"
        f"DATASET: {params.get('dataset')}\n"
        f"SEED: {params.get('seed')}\n"
        f"DURATION (s): {result.get('duration_seconds', 0):.1f}\n"
        f"TOTAL PARAMS: {result.get('params_total', 0):,}\n"
        f"TRAINABLE: {result.get('params_trainable', 0):,}\n"
        f"GPU PEAK MB: {result.get('gpu_peak_mb', 0):.0f}\n"
        f"TEST_LOSS: {final.get('test_loss', float('nan')):.4f}\n"
        f"TEST_PPL:  {final.get('test_ppl', float('nan')):.4f}\n"
        f"TEST_BPC:  {final.get('test_bpc', float('nan')):.4f}\n"
    )
    (run_dir / "final_summary.txt").write_text(summary, encoding="utf-8")


def append_to_index(results_root: str | Path, params: Dict[str, Any], result: Dict[str, Any], run_dir: Path) -> Path:
    """results/_index.csv'ye tek satır ekler. Yoksa header'la birlikte oluşturur."""
    path = Path(results_root) / "_index.csv"
    header = [
        "run_id", "method", "run_name", "dataset", "seed",
        "test_loss", "test_ppl", "test_bpc",
        "duration_s", "params_total", "params_trainable", "gpu_peak_mb",
        "num_epochs", "batch_size", "learning_rate", "max_seq_len",
        "started_at", "run_dir",
    ]
    final = result.get("final_metrics", {})
    row = [
        run_dir.name,
        params.get("method_name", ""),
        params.get("run_name", ""),
        params.get("dataset", ""),
        params.get("seed", ""),
        final.get("test_loss", ""),
        final.get("test_ppl", ""),
        final.get("test_bpc", ""),
        result.get("duration_seconds", ""),
        result.get("params_total", ""),
        result.get("params_trainable", ""),
        result.get("gpu_peak_mb", ""),
        params.get("num_epochs", ""),
        params.get("batch_size", ""),
        params.get("learning_rate", ""),
        params.get("max_seq_len", ""),
        result.get("started_at", ""),
        str(run_dir),
    ]
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)
    return path


# ─────────────────────────────────────────────────────────
# Grafikler
# ─────────────────────────────────────────────────────────
def make_plots(run_dir: Path, result: Dict[str, Any]) -> List[Path]:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    history = result.get("history", {})
    epoch_hist = history.get("epoch", [])
    step_hist = history.get("step", [])

    paths = []

    # Loss curve (epoch)
    if epoch_hist:
        fig, ax = plt.subplots(figsize=(9, 5))
        ep = [e["epoch"] for e in epoch_hist]
        ax.plot(ep, [e["train_loss"] for e in epoch_hist], "-o", label="train", color="#3b82f6")
        ax.plot(ep, [e["val_loss"]   for e in epoch_hist], "-s", label="val",   color="#ef4444")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss curve"); ax.legend()
        ax.grid(alpha=0.3)
        p = plots_dir / "loss_curve.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        paths.append(p)

        # PPL curve
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(ep, [e["val_ppl"] for e in epoch_hist], "-o", color="#1d4ed8")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Val PPL"); ax.set_title("Validation perplexity")
        ax.grid(alpha=0.3)
        p = plots_dir / "ppl_curve.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        paths.append(p)

    # Stability (step)
    if step_hist:
        fig, ax = plt.subplots(figsize=(10, 4))
        steps = [s["step"] for s in step_hist]
        losses = [s["train_loss"] for s in step_hist]
        ax.plot(steps, losses, alpha=0.4, label="raw", color="#94a3b8")
        if len(losses) >= 10:
            window = max(3, len(losses) // 20)
            ma = _moving_average(losses, window)
            ax.plot(steps, ma, color="#dc2626", label=f"{window}-step MA")
        ax.set_xlabel("Step"); ax.set_ylabel("Train loss"); ax.set_title("Training stability"); ax.legend()
        ax.grid(alpha=0.3)
        p = plots_dir / "stability.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        paths.append(p)

    return paths


def _moving_average(xs: List[float], window: int) -> List[float]:
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        seg = xs[lo : i + 1]
        out.append(sum(seg) / len(seg))
    return out


# ─────────────────────────────────────────────────────────
# Log handler — list-backed
# ─────────────────────────────────────────────────────────
class ListLogger:
    """Hem print eder hem listede tutar — sonra save_results'a verilir."""

    def __init__(self):
        self.lines: List[str] = []

    def __call__(self, msg: str) -> None:
        stamp = datetime.utcnow().strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        self.lines.append(line)
        print(line)


def run_pipeline(model: nn.Module, params: Dict[str, Any], adapter: Optional[Any] = None) -> Tuple[Path, Dict[str, Any]]:
    """Notebook'taki 'çalıştır' hücresinin tek satırlık özeti.

    Args:
        model: GPT2Wrapper veya GPT2LMHeadModel
        params: tüm PARAMS dict
        adapter: opsiyonel Yön B adapter

    Returns:
        (run_dir, result_dict)
    """
    run_dir = make_run_dir(
        results_root=params.get("results_root", "./results"),
        method=params.get("method_name", "unknown"),
        run_name=params.get("run_name", "run"),
    )
    logger = ListLogger()
    logger(f"Run dir: {run_dir}")
    logger(f"Device: {get_device()} | GPU: {gpu_info()['name']}")
    logger(f"Method params: {json.dumps({k:v for k,v in params.items() if k.startswith(_method_prefix(params))}, default=str)}")

    started_at = datetime.utcnow().isoformat()
    trainer = StandaloneTrainer(model, params, run_dir, log_fn=logger, adapter=adapter)
    result = trainer.fit()
    result["started_at"] = started_at
    result["device"] = str(get_device())

    save_results(run_dir, params, result, logger.lines)
    append_to_index(params.get("results_root", "./results"), params, result, run_dir)
    plots = make_plots(run_dir, result)
    logger(f"Saved {len(plots)} plots → {run_dir / 'plots'}")
    return run_dir, result


def _method_prefix(params: Dict[str, Any]) -> str:
    m = params.get("method_name", "")
    if m == "baseline":
        return "baseline_"
    if m.startswith("A1"): return "pcr_"
    if m.startswith("A2"): return "li_"
    if m.startswith("A3"): return "rbr_"
    if m.startswith("B1"): return "mi_"
    if m.startswith("B2"): return "cma_"
    if m.startswith("B3"): return "fw_"
    if m.startswith("C1"): return "pc_"
    if m.startswith("C2"): return "ep_"
    if m.startswith("C3"): return "cha_"
    return ""
