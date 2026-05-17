"""C1 — Predictive Coding Unified Network.

Aynı tahmin hatası (ε) hem routing skoru hem yerel ağırlık güncellemesi.
Prototip: attention katmanına PC routing eklenir; yerel update gradient
flag'iyle düzenlenir — saf "backprop yok" iddiası tam Faz 2 hedefi, prototipte
mixed (PRD'de pc_use_local_only=True).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attention.base_router import BaseAttentionRouter


class PCUnifiedAttention(BaseAttentionRouter):
    def __init__(self, exp_config, gpt2_config):
        super().__init__(exp_config, gpt2_config)
        mp = exp_config.method_params
        self.delta = float(mp.get("pc_delta", 1e-6))
        self.precision_init = float(mp.get("pc_precision_init", 1.0))
        self.update_lr = float(mp.get("pc_update_lr", 0.01))
        self.use_local_only = bool(mp.get("pc_use_local_only", False))

        # Bir sonraki katmanı tahmin eden predictor (basit linear)
        self.layer_predictor = nn.Linear(self.embed_dim, self.embed_dim)
        # Precision — log-uzayda öğrenilir
        self.log_precision = nn.Parameter(torch.tensor(0.0))

    def _layer_error(self, hidden: torch.Tensor) -> torch.Tensor:
        """ε = hidden - f_θ(hidden)  (self-prediction proxy — gerçek PC'de
        l+1 katmanı kullanılır; prototip uyarlama)."""
        return hidden - self.layer_predictor(hidden)

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = self._layer_error(hidden_states)              # [B, S, H]
        err_norm = eps.norm(dim=-1)                          # [B, S]
        # Score(i, j) = 1 / (||ε_j|| + δ) — j ne kadar tahminden sapıyorsa
        # o kadar çok bakılması gereken token
        score_per_j = 1.0 / (err_norm + self.delta)          # [B, S]
        scores = score_per_j.unsqueeze(1).expand(-1, q.size(-2), -1)  # [B, S_i, S_j]
        scores = scores.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # head broadcast

        causal = self._causal_mask(q.size(-2), k.size(-2), q.device)
        scores = scores.masked_fill(~causal, float("-inf"))
        if attention_mask is not None:
            scores = scores + attention_mask

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        out = torch.matmul(weights, v)

        # Yerel update — sadece predictor ağırlıklarına etki eder
        if self.training and self.use_local_only:
            with torch.enable_grad():
                precision = torch.exp(self.log_precision)
                local_loss = 0.5 * precision * (eps ** 2).sum()
                # Sadece predictor + log_precision için
                params = list(self.layer_predictor.parameters()) + [self.log_precision]
                grads = torch.autograd.grad(
                    local_loss, params, retain_graph=True, allow_unused=True
                )
                for p, g in zip(params, grads):
                    if g is not None:
                        p.data -= self.update_lr * g

        self._last_stats = {
            "err_mean": err_norm.mean().item(),
            "precision": float(torch.exp(self.log_precision).item()),
        }
        return out, weights
