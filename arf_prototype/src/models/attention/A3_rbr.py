"""A3 — Resonance-Based Router (ART-style).

Memory bank slotları içerisinden vigilance eşiğini geçenler aktive olur.
Mismatch durumunda yeni slot açılır (PRD prototip: en az kullanılan slot
overwrite edilir — bellek sabit kalır).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .base_router import BaseAttentionRouter


class ResonanceBasedRouter(BaseAttentionRouter):
    def __init__(self, exp_config, gpt2_config):
        super().__init__(exp_config, gpt2_config)
        mp = exp_config.method_params
        self.vigilance = float(mp.get("rbr_vigilance", 0.7))
        self.num_slots = int(mp.get("rbr_num_slots", 64))
        self.slot_dim = int(mp.get("rbr_slot_dim", self.embed_dim))
        self.beta = float(mp.get("rbr_beta", 1e-6))
        self.slot_lr = float(mp.get("rbr_slot_lr", 0.1))

        # Memory bank — slot başına bir prototype vektör
        self.memory_bank = nn.Parameter(torch.randn(self.num_slots, self.slot_dim) * 0.02)
        # Slot kullanım sayacı (yeni slot atama için)
        self.register_buffer("slot_usage", torch.zeros(self.num_slots))

        # Memory bank'tan value'ya projeksiyon
        self.value_proj = nn.Linear(self.slot_dim, self.embed_dim)

    def _resonance(self, hidden: torch.Tensor) -> torch.Tensor:
        """e_i · m_j / (||m_j|| + β). hidden: [B, S, H], mem: [K, H]
        Returns: [B, S, K]
        """
        m_norm = self.memory_bank / (self.memory_bank.norm(dim=-1, keepdim=True) + self.beta)
        return torch.einsum("bsh,kh->bsk", hidden, m_norm)

    def _decide(self, res: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vigilance üstü slot maskesi + normalize."""
        mask = res >= self.vigilance
        masked = res * mask.float()
        row_sum = masked.sum(-1, keepdim=True).clamp(min=1e-8)
        weights = masked / row_sum                    # [B, S, K]
        mismatch = mask.sum(-1) == 0                  # [B, S]
        return weights, mismatch

    def _maybe_open_slot(self, hidden: torch.Tensor, mismatch: torch.Tensor) -> None:
        """Mismatch tokenları için en az kullanılan slotu overwrite et."""
        if not self.training or not mismatch.any():
            return
        with torch.no_grad():
            # Tokens that didn't match — mean ile özet
            new_proto = hidden[mismatch].mean(0)       # [H]
            target_slot = int(self.slot_usage.argmin().item())
            self.memory_bank.data[target_slot] = (
                (1 - self.slot_lr) * self.memory_bank.data[target_slot]
                + self.slot_lr * new_proto
            )
            self.slot_usage[target_slot] = 0
            self.slot_usage += 1  # tüm slotlar yaşlanır

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Token başına memory bank routing
        res = self._resonance(hidden_states)            # [B, S, K]
        weights, mismatch = self._decide(res)
        self._maybe_open_slot(hidden_states.detach(), mismatch)

        # Output: weighted sum over memory_bank, then project to embed_dim
        mem_out = torch.einsum("bsk,kh->bsh", weights, self.memory_bank)   # [B, S, H]
        out = self.value_proj(mem_out)
        # Head-wise format'a uydur — gerçek attention output gibi davranır
        out = self._split_heads(out)                    # [B, H, S, d]

        # attn_weights çıktısı için — [B, H, S, S] yerine [B, H, S, K]
        # ama interface tutarlılığı için S, S beklenebilir; biz [B, 1, S, K] döneriz
        pseudo_weights = weights.unsqueeze(1)           # [B, 1, S, K]
        self._last_stats = {
            "mismatch_rate": mismatch.float().mean().item(),
            "active_slots":  (weights.sum(0).sum(0) > 0).sum().item(),
        }
        return out, pseudo_weights
