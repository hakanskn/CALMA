"""A2 — Lateral Inhibition Sparse Router.

Wilson-Cowan iteratif yaklaşımı + Hebbian inhibition matrisi.
Sparse aktivasyon ReLU ile üretilir; softmax dağıtımı yerine gerçek bastırma.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .base_router import BaseAttentionRouter


class LateralInhibitionRouter(BaseAttentionRouter):
    def __init__(self, exp_config, gpt2_config):
        super().__init__(exp_config, gpt2_config)
        mp = exp_config.method_params
        self.lam = float(mp.get("li_lambda", 0.5))
        self.num_iters = int(mp.get("li_num_iters", 3))
        self.hebbian_lr = float(mp.get("li_hebbian_lr", 0.01))
        self.weight_decay_rate = float(mp.get("li_weight_decay", 0.001))
        self.min_activation = float(mp.get("li_min_activation", 0.01))

        # Inhibition matrix — seq-len bazlı. Maks pozisyon kadar
        max_pos = gpt2_config.n_positions
        self.W_inhib = nn.Parameter(torch.randn(max_pos, max_pos) * 0.01)

    def _inhibit(self, scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Wilson-Cowan single/iterative step. scores: [B, H, S, S]."""
        W = self.W_inhib[:seq_len, :seq_len]   # [S, S]
        alpha = scores
        for _ in range(self.num_iters):
            inhib = self.lam * torch.matmul(alpha, W.T)
            alpha = torch.relu(scores - inhib)
            alpha = alpha.clamp(min=self.min_activation)
        return alpha

    def _hebbian_update(self, alpha: torch.Tensor, seq_len: int) -> None:
        """Eğitim sırasında çağrılır — gradient-free."""
        if not self.training:
            return
        with torch.no_grad():
            # alpha: [B, H, S, S] → [S, S]
            mean_alpha = alpha.mean(dim=(0, 1))           # [S, S]
            corr = mean_alpha.T @ mean_alpha               # [S, S]
            dW = self.hebbian_lr * corr - self.weight_decay_rate * self.W_inhib[:seq_len, :seq_len]
            self.W_inhib.data[:seq_len, :seq_len] += dW

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = torch.matmul(q, k.transpose(-1, -2)) * self.scale     # [B, H, S, S]

        causal = self._causal_mask(q.size(-2), k.size(-2), q.device)
        raw = raw.masked_fill(~causal, 0.0)   # softmax değil — sparse

        alpha = self._inhibit(raw, raw.size(-1))
        # Causal yapı korunsun
        alpha = alpha.masked_fill(~causal, 0.0)
        # Row normalize — value ile çarpımı dengele
        alpha = alpha / (alpha.sum(-1, keepdim=True) + 1e-9)

        self._hebbian_update(alpha.detach(), alpha.size(-1))

        alpha_drop = self.attn_dropout(alpha)
        out = torch.matmul(alpha_drop, v)
        self._last_stats = {
            "sparsity": (alpha < self.min_activation * 1.01).float().mean().item(),
            "W_norm":   self.W_inhib.norm().item(),
        }
        return out, alpha
