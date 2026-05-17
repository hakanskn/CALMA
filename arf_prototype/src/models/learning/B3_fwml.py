"""B3 — Fast Weight Meta-Learning.

Slow weights donar, her attention katmanı üstüne LoRA-rank fast weight A_t eklenir.
Güncelleme Hebbian outer product — gradient-free, tamamen yerel.

Prototip uygulama: PyTorch hook ile her transformer bloğunun çıktısına A_t · x ekler.
Snapshot/reset session bazlı yapılır.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .base_adapter import BaseAdaptationMethod


class FastWeightMetaLearning(BaseAdaptationMethod):
    def __init__(self, model: nn.Module, exp_config):
        super().__init__(model, exp_config)
        mp = exp_config.method_params
        self.rank = int(mp.get("fw_rank", 32))
        self.hebbian_lr = float(mp.get("fw_hebbian_lr", 0.01))
        self.decay = float(mp.get("fw_decay", 0.99))
        self.update_rule = str(mp.get("fw_update_rule", "hebbian"))
        self.reset_session = bool(mp.get("fw_reset_session", True))
        self.freeze_slow = bool(mp.get("fw_freeze_slow", True))

        hidden = exp_config.model_hidden_size
        n_layers = exp_config.model_num_layers
        device = next(model.parameters()).device

        # LoRA-rank fast weights: A_t = U_t @ V_t  → [d, r] × [r, d]
        self.fast_U: List[nn.Parameter] = [
            nn.Parameter(torch.zeros(hidden, self.rank, device=device)) for _ in range(n_layers)
        ]
        self.fast_V: List[nn.Parameter] = [
            nn.Parameter(torch.zeros(self.rank, hidden, device=device)) for _ in range(n_layers)
        ]
        for p in self.fast_U + self.fast_V:
            p.requires_grad = False  # gradient-free

        if self.freeze_slow:
            for p in self.model.parameters():
                p.requires_grad = False

        # Hook'lar: her bloğun çıktısına fast contribution ekle
        self._handles = []
        self._last_inputs: Dict[int, torch.Tensor] = {}
        for idx, block in enumerate(self.model.model.transformer.h):
            self._handles.append(block.register_forward_hook(self._make_hook(idx)))

        self.outer_optimizer = None  # gradient-free

    # ─────────────────────────────────────────────────────
    def _make_hook(self, layer_idx: int):
        def hook(_module, inputs, output):
            x = inputs[0]
            self._last_inputs[layer_idx] = x.detach()
            U = self.fast_U[layer_idx]
            V = self.fast_V[layer_idx]
            # h = x · (U V)^T   ≡ (x · V^T) · U^T
            fast_out = (x @ V.T) @ U.T
            if isinstance(output, tuple):
                return (output[0] + fast_out,) + output[1:]
            return output + fast_out
        return hook

    # ─────────────────────────────────────────────────────
    @property
    def requires_meta_loop(self) -> bool:
        return False

    def adapt(self, context_batch: Dict[str, torch.Tensor]) -> None:
        """Forward pass → Hebbian outer-product güncelleme."""
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            self.model(
                input_ids=context_batch["input_ids"],
                attention_mask=context_batch.get("attention_mask"),
            )
            # Each layer'da kayıtlı son giriş ile Hebbian update
            for idx, x in self._last_inputs.items():
                # x: [B, S, H]   →  outer = x^T · x ≈ [H, H]
                # LoRA-rank: U += η · (mean_x · pca_dir);  basit prototip:
                mean_h = x.mean(dim=(0, 1))                       # [H]
                key = mean_h[: self.rank]                          # crude — [r]
                # Hebbian: U += η · h ⊗ k
                self.fast_U[idx].data += self.hebbian_lr * torch.outer(mean_h, key)
                # Decay
                self.fast_U[idx].data *= self.decay
                self.fast_V[idx].data *= self.decay
        if was_training:
            self.model.train()

    def meta_train_step(self, support_batch, query_batch):
        # B3'te outer loop yok — gradient-free
        return 0.0

    def reset(self) -> None:
        for U, V in zip(self.fast_U, self.fast_V):
            nn.init.zeros_(U)
            nn.init.zeros_(V)
        self._last_inputs.clear()

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
