"""Utility fonksiyonlar — seed yönetimi, GPU bilgisi, Drive senkronizasyonu, timing."""

from __future__ import annotations

import os
import random
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

# Varsayılan seed seti — base_config.yaml ile senkron
SEEDS = [42, 123, 456, 789, 1024]


def seed_everything(seed: int = 42) -> None:
    """Tüm RNG'leri sabitle. Reproducibility için."""
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


def gpu_name() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    return torch.cuda.get_device_name(0)


def gpu_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def reset_gpu_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@contextmanager
def timer(name: str = "") -> Iterator[dict]:
    """Block süresini ölç. with timer() as t: ...   t['elapsed']"""
    state = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield state
    finally:
        state["elapsed"] = time.perf_counter() - start


def count_parameters(model: torch.nn.Module, only_trainable: bool = True) -> int:
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_copy(src: str | Path, dst: str | Path) -> None:
    """Drive'a sync için güvenli kopyalama — kısmi yazımı engelle."""
    src, dst = Path(src), Path(dst)
    ensure_dir(dst.parent)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def colab_drive_mounted(drive_base: Optional[str] = None) -> bool:
    """Drive'ın bağlı olduğunu doğrula."""
    if drive_base is None:
        drive_base = "/content/drive/MyDrive"
    return os.path.exists(drive_base)


def estimate_remaining_seconds(step: int, total_steps: int, started: float) -> float:
    if step <= 0:
        return -1.0
    elapsed = time.perf_counter() - started
    rate = step / elapsed
    remaining = (total_steps - step) / max(rate, 1e-9)
    return remaining


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"
