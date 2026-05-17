"""C3 — Contrastive Hebbian Attention.

Hinton'un Forward-Forward fikrinin attention versiyonu.
Routing: ΔW = h⁺h⁺ᵀ − h⁻h⁻ᵀ — h⁺ kenetlenmiş, h⁻ serbest faz.
Prototip: iki forward yerine dropout-varyasyonu ile pozitif/negatif view.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attention.base_router import BaseAttentionRouter


class ContrastiveHebbianAttention(BaseAttentionRouter):
    def __init__(self, exp_config, gpt2_config):
        super().__init__(exp_config, gpt2_config)
        mp = exp_config.method_params
        self.hebb_lr = float(mp.get("cha_hebb_lr", 0.01))
        self.neg_ratio = float(mp.get("cha_neg_ratio", 1.0))
        self.temperature = float(mp.get("cha_temperature", 1.0))
        self.clamp_strength = float(mp.get("cha_clamp_strength", 0.1))

    def _scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return torch.matmul(q, k.transpose(-1, -2)) * self.scale

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # h⁻: serbest faz (dropout aktif iken normal forward)
        s_minus = self._scores(q, k)                            # [B, H, S, S]

        # h⁺: kenetlenmiş — küçük perturb ile pozitif görünüm
        noise = torch.randn_like(hidden_states) * self.clamp_strength * hidden_states.std()
        clamped = hidden_states + noise
        qkv_p = self.c_attn(clamped)
        qp, kp, _ = qkv_p.split(self.embed_dim, dim=-1)
        qp = self._split_heads(qp)
        kp = self._split_heads(kp)
        s_plus = self._scores(qp, kp)

        # Contrastive routing skoru
        delta = s_plus - self.neg_ratio * s_minus
        delta = delta / self.temperature

        causal = self._causal_mask(q.size(-2), k.size(-2), q.device)
        delta = delta.masked_fill(~causal, float("-inf"))
        if attention_mask is not None:
            delta = delta + attention_mask

        weights = F.softmax(F.relu(delta), dim=-1).clamp(min=1e-9)
        weights = weights / weights.sum(-1, keepdim=True)
        weights = self.attn_dropout(weights)
        out = torch.matmul(weights, v)

        # Lokal Hebbian update — sadece c_attn ağırlıklarının K bloğu
        if self.training:
            with torch.no_grad():
                # h⁺ ve h⁻ aktivasyonlarının korelasyon farkı
                hp = clamped.mean(0)                            # [S, H]
                hm = hidden_states.mean(0)
                dW = self.hebb_lr * (hp.T @ hp - hm.T @ hm) / max(1, hp.size(0))
                # c_attn weight shape: [3H, H]; key block [H:2H]
                kw = self.c_attn.weight.data[self.embed_dim : 2 * self.embed_dim]
                kw += dW[: kw.size(0), : kw.size(1)]
                self.c_attn.weight.data[self.embed_dim : 2 * self.embed_dim] = kw

        self._last_stats = {
            "delta_mean": (s_plus - s_minus).abs().mean().item(),
        }
        return out, weights
