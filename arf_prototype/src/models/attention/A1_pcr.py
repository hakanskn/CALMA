"""A1 — Predictive Coding Router.

Routing skoru: score(i, j) = 1 / (||x_j - f_θ(x_i)||² + δ)

Bellek için: predictor head-aware değildir; tüm head'ler aynı predictor'ı paylaşır
(prototip seçimi; PRD'de açık). MLP boyutu config'ten alınır.

Causal mask uygulanır — GPT-2 decoder mantığı.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_router import BaseAttentionRouter


class PredictiveCodingRouter(BaseAttentionRouter):
    """Score = 1 / (prediction_error + δ)."""

    def __init__(self, exp_config, gpt2_config):
        super().__init__(exp_config, gpt2_config)
        mp = exp_config.method_params
        pred_dim = int(mp.get("pcr_predictor_dim", 256))
        n_layers = int(mp.get("pcr_predictor_layers", 2))
        self.delta = float(mp.get("pcr_delta", 1e-6))
        self.temperature = float(mp.get("pcr_temperature", 1.0))

        layers: list[nn.Module] = []
        in_d = self.embed_dim
        for _ in range(max(1, n_layers - 1)):
            layers += [nn.Linear(in_d, pred_dim), nn.ReLU()]
            in_d = pred_dim
        layers.append(nn.Linear(in_d, self.embed_dim))
        self.predictor = nn.Sequential(*layers)

    # ─────────────────────────────────────────────────────
    def _prediction_error(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        hidden: [B, S, H]
        Returns: errors [B, S_i, S_j] — token i'nin token j'yi tahmin hatası
        """
        pred = self.predictor(hidden)                  # [B, S, H]
        # Pairwise MSE: ||x_j - pred_i||²
        diff = hidden.unsqueeze(1) - pred.unsqueeze(2)  # [B, S_i, S_j, H]
        return (diff * diff).sum(-1)                    # [B, S_i, S_j]

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Tek skor map'i tüm head'lere broadcast edilir
        errors = self._prediction_error(hidden_states)         # [B, S, S]
        scores = 1.0 / (errors + self.delta)
        scores = scores / self.temperature

        # [B, 1, S, S] -> head'lere kopyala
        scores = scores.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        causal = self._causal_mask(q.size(-2), k.size(-2), q.device)
        scores = scores.masked_fill(~causal, float("-inf"))
        if attention_mask is not None:
            scores = scores + attention_mask

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        out = torch.matmul(weights, v)
        self._last_stats = {
            "mean_err": errors.mean().item(),
            "max_err":  errors.max().item(),
        }
        return out, weights
