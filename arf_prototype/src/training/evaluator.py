"""Evaluator — perplexity, loss, BPC hesaplama."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader

from ..utils import get_device


@dataclass
class EvalMetrics:
    loss: float
    perplexity: float
    bits_per_char: float
    num_tokens: int

    def to_dict(self) -> dict:
        return {
            "loss": self.loss,
            "perplexity": self.perplexity,
            "bits_per_char": self.bits_per_char,
            "num_tokens": self.num_tokens,
        }


class Evaluator:
    def __init__(self, model, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or get_device()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, max_batches: Optional[int] = None) -> EvalMetrics:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
            )
            ntok = batch["labels"].ne(-100).sum().item()
            total_loss += out.loss.item() * ntok
            total_tokens += ntok

        avg_loss = total_loss / max(1, total_tokens)
        ppl = math.exp(min(20.0, avg_loss))   # clip overflow için
        bpc = avg_loss / math.log(2)
        return EvalMetrics(
            loss=float(avg_loss),
            perplexity=float(ppl),
            bits_per_char=float(bpc),
            num_tokens=int(total_tokens),
        )
