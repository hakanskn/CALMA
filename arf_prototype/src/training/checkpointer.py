"""Checkpoint kayıt/yükleme — Drive'a periyodik yazım, session resume için."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..utils import ensure_dir


class Checkpointer:
    def __init__(self, exp_config):
        self.config = exp_config
        self.dir = ensure_dir(exp_config.checkpoint_dir())
        self.keep_last = exp_config.keep_last_n_checkpoints

    def _list(self) -> list[str]:
        return sorted(glob.glob(str(self.dir / "ckpt_step_*.pt")))

    def _cleanup(self) -> None:
        files = self._list()
        if len(files) <= self.keep_last:
            return
        for f in files[: -self.keep_last]:
            try:
                os.remove(f)
            except OSError:
                pass

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        step: int,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        ckpt = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "metrics": metrics or {},
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        path = self.dir / f"ckpt_step_{step:06d}.pt"
        torch.save(ckpt, path)
        self._cleanup()
        return str(path)

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> int:
        files = self._list()
        if not files:
            return 0
        ckpt = torch.load(files[-1], map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        torch.set_rng_state(ckpt["rng_state"])
        if torch.cuda.is_available() and ckpt.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        return int(ckpt.get("step", 0))
