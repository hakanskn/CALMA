"""Standart scaled dot-product attention — sanity-check / re-implement baseline.

Pratikte baseline yöntemi mevcut HF GPT2Attention'ı kullanır (inject yok).
Bu modül, eğer bir gün baseline'ı yeniden implement etmek istersek diye burada.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .base_router import BaseAttentionRouter


class StandardAttention(BaseAttentionRouter):
    """Klasik softmax(QKᵀ/√d) attention — referans."""

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [b,h,s,s]

        causal = self._causal_mask(q.size(-2), k.size(-2), q.device)
        scores = scores.masked_fill(~causal, float("-inf"))
        if attention_mask is not None:
            scores = scores + attention_mask

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        out = torch.matmul(weights, v)
        self._last_stats = {"mean_entropy": _entropy(weights).item()}
        return out, weights


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * (p.clamp_min(1e-9)).log()).sum(-1).mean()
