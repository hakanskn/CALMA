"""C2 — Equilibrium Propagation Transformer (prototip yaklaşım).

Tam EP iteratif minimizasyon GPT-2 ölçeğinde çok pahalı; PRD'de
`ep_use_simplified=True` ile basitleştirilmiş bir uyarlama tanımlandı:

Prototip:
  - Serbest faz: standart attention (h_minus).
  - Kenetlenmiş faz: self-supervised hedefe doğru hafif itki (small β).
  - Routing skor: cross(h_plus, h_minus) farkından türetilir.
  - Loss/update yine standart cross-entropy (yapısal gradient) ile;
    EP-style local update Faz 2'de iteratif hale getirilecek.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attention.base_router import BaseAttentionRouter


class EquilibriumPropAttention(BaseAttentionRouter):
    def __init__(self, exp_config, gpt2_config):
        super().__init__(exp_config, gpt2_config)
        mp = exp_config.method_params
        self.beta = float(mp.get("ep_nudge_beta", 0.05))
        self.free_iters = int(mp.get("ep_free_iters", 10))
        self.clamped_iters = int(mp.get("ep_clamped_iters", 4))
        self.energy_lr = float(mp.get("ep_energy_lr", 0.01))
        self.use_simplified = bool(mp.get("ep_use_simplified", True))

        # Self-supervised hedef projeksiyonu (kenetlenmiş faz)
        self.target_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _phase_scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Standart scaled dot-product — bir fazın skorları."""
        return torch.matmul(q, k.transpose(-1, -2)) * self.scale

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        free_scores = self._phase_scores(q, k)
        if self.use_simplified:
            # Kenetlenmiş: hidden'i target'a doğru hafifçe nudge et
            target = self.target_proj(hidden_states)
            clamped_h = hidden_states + self.beta * (target - hidden_states)
            qkv_c = self.c_attn(clamped_h)
            qc, kc, _ = qkv_c.split(self.embed_dim, dim=-1)
            qc = self._split_heads(qc)
            kc = self._split_heads(kc)
            clamped_scores = self._phase_scores(qc, kc)
        else:
            clamped_scores = free_scores

        # EP routing: ortalama (h+ ve h-) — fark gradient sinyali olur
        combined = 0.5 * (free_scores + clamped_scores)

        causal = self._causal_mask(q.size(-2), k.size(-2), q.device)
        combined = combined.masked_fill(~causal, float("-inf"))
        if attention_mask is not None:
            combined = combined + attention_mask

        weights = F.softmax(combined, dim=-1)
        weights = self.attn_dropout(weights)
        out = torch.matmul(weights, v)

        self._last_stats = {
            "energy_diff": (clamped_scores - free_scores).abs().mean().item(),
        }
        return out, weights
