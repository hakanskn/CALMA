"""Yön A için ortak attention router arayüzü.

HuggingFace `GPT2Attention.forward` signature'ına uyar:

    forward(hidden_states, layer_past=None, attention_mask=None, head_mask=None,
            encoder_hidden_states=None, encoder_attention_mask=None,
            use_cache=False, output_attentions=False)
    -> (attn_output, present, [attn_weights])

Her alt sınıf `_compute_attention(hidden, causal_mask)` implement eder.
Kayıt: son `attn_weights` ve method-specific istatistikler `self._last_stats`
sözlüğünde tutulur — trainer/eval bu sözlüğü okur.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class BaseAttentionRouter(nn.Module):
    """Ortak skeleton — Q/K/V projections, causal mask, output projection."""

    def __init__(self, exp_config, gpt2_config):
        super().__init__()
        self.exp_config = exp_config
        self.method_params = exp_config.method_params
        self.embed_dim = gpt2_config.n_embd
        self.num_heads = gpt2_config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Standart projections — yöntemler kendi Q/K/V'lerini ekleyebilir
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.attn_dropout = nn.Dropout(gpt2_config.attn_pdrop)
        self.resid_dropout = nn.Dropout(gpt2_config.resid_pdrop)

        # Causal mask buffer — uzun seq için lazy genişler
        max_pos = gpt2_config.n_positions
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_pos, max_pos), dtype=torch.bool)).view(
                1, 1, max_pos, max_pos
            ),
            persistent=False,
        )

        self._last_stats: dict = {}

    # ─────────────────────────────────────────────────────
    # Yardımcı: Q/K/V split + head reshape
    # ─────────────────────────────────────────────────────
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.size()
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)  # [b, h, s, d]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, s, d = x.size()
        return x.transpose(1, 2).contiguous().view(b, s, h * d)

    def _qkv(self, hidden_states: torch.Tensor):
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.embed_dim, dim=-1)
        return self._split_heads(q), self._split_heads(k), self._split_heads(v)

    def _causal_mask(self, q_len: int, k_len: int, device) -> torch.Tensor:
        return self.bias[:, :, k_len - q_len : k_len, :k_len]  # type: ignore[attr-defined]

    # ─────────────────────────────────────────────────────
    # Alt sınıfın implement edeceği skor fonksiyonu
    # ─────────────────────────────────────────────────────
    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (attn_output [b,h,s,d], attn_weights [b,h,s,s])."""
        raise NotImplementedError

    # ─────────────────────────────────────────────────────
    # GPT2Attention-compat forward
    # ─────────────────────────────────────────────────────
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past=None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        q, k, v = self._qkv(hidden_states)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        present = (k, v) if use_cache else None

        attn_out, attn_weights = self._compute_attention(
            q, k, v, hidden_states, attention_mask
        )
        attn_out = self._merge_heads(attn_out)
        attn_out = self.c_proj(attn_out)
        attn_out = self.resid_dropout(attn_out)

        outputs = (attn_out, present)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs
