"""B2 — Contrastive Meta-Adaptation.

NT-Xent contrastive loss (SimCLR) + meta-training across domains.
Pozitif çift: encoder dropout ile iki farklı görünüm (ESimCSE tarzı).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_adapter import BaseAdaptationMethod


class ContrastiveMetaAdaptation(BaseAdaptationMethod):
    def __init__(self, model: nn.Module, exp_config):
        super().__init__(model, exp_config)
        mp = exp_config.method_params
        self.temperature = float(mp.get("cma_temperature", 0.07))
        self.inner_lr = float(mp.get("cma_inner_lr", 1e-4))
        self.inner_steps = int(mp.get("cma_inner_steps", 3))
        self.outer_lr = float(mp.get("cma_outer_lr", 5e-5))
        self.projection_dim = int(mp.get("cma_projection_dim", 128))

        hidden = exp_config.model_hidden_size
        self.projector = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.projection_dim),
        ).to(next(model.parameters()).device)

        self.outer_optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.projector.parameters()),
            lr=self.outer_lr,
        )

    @property
    def requires_meta_loop(self) -> bool:
        return True

    # ─────────────────────────────────────────────────────
    def _encode_views(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """İki dropout-li forward → iki proje. Returns: (z1, z2) [B, D]."""
        self.model.train()  # dropout aktif olsun
        out1 = self.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"), output_hidden_states=True)
        out2 = self.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"), output_hidden_states=True)
        # Mean-pool over sequence
        h1 = out1.hidden_states[-1].mean(dim=1)   # [B, H]
        h2 = out2.hidden_states[-1].mean(dim=1)
        z1 = F.normalize(self.projector(h1), dim=-1)
        z2 = F.normalize(self.projector(h2), dim=-1)
        return z1, z2

    def _nt_xent(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        bs = z1.size(0)
        z = torch.cat([z1, z2], dim=0)                                # [2B, D]
        sim = (z @ z.T) / self.temperature
        # Diagonal
        mask = torch.eye(2 * bs, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, float("-inf"))
        # Pozitif çiftler: (i, i+B) ve (i+B, i)
        labels = torch.cat([
            torch.arange(bs, device=z.device) + bs,
            torch.arange(bs, device=z.device),
        ])
        return F.cross_entropy(sim, labels)

    # ─────────────────────────────────────────────────────
    def adapt(self, context_batch: Dict[str, torch.Tensor]) -> None:
        was_training = self.model.training
        inner_opt = torch.optim.SGD(
            list(self.model.parameters()) + list(self.projector.parameters()),
            lr=self.inner_lr,
        )
        for _ in range(self.inner_steps):
            z1, z2 = self._encode_views(context_batch)
            loss = self._nt_xent(z1, z2)
            inner_opt.zero_grad()
            loss.backward()
            inner_opt.step()
        if not was_training:
            self.model.eval()

    def meta_train_step(self, support_batch: Dict[str, torch.Tensor], query_batch: Dict[str, torch.Tensor]) -> float:
        original = {n: p.data.clone() for n, p in self.model.named_parameters()}
        proj_original = {n: p.data.clone() for n, p in self.projector.named_parameters()}
        self.adapt(support_batch)

        z1, z2 = self._encode_views(query_batch)
        query_loss = self._nt_xent(z1, z2)

        self.outer_optimizer.zero_grad()
        query_loss.backward()
        # Orijinal'e geri sar, sonra outer step
        for n, p in self.model.named_parameters():
            p.data.copy_(original[n])
        for n, p in self.projector.named_parameters():
            p.data.copy_(proj_original[n])
        self.outer_optimizer.step()
        return float(query_loss.item())

    def reset(self) -> None:
        pass
