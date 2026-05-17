"""B1 — Meta-Initialized Self-Supervised Test-Time Training.

FOMAML çatısı + self-supervised inner loss (masked LM).
GPT-2 üzerinde maskeleme için EOS token kullanılır (BERT mask_token yok).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base_adapter import BaseAdaptationMethod
from ...data.preprocessor import mask_tokens


class MISSelfSupervisedTTT(BaseAdaptationMethod):
    def __init__(self, model: nn.Module, exp_config):
        super().__init__(model, exp_config)
        mp = exp_config.method_params
        self.inner_lr = float(mp.get("mi_inner_lr", 1e-4))
        self.inner_steps = int(mp.get("mi_inner_steps", 5))
        self.outer_lr = float(mp.get("mi_outer_lr", 5e-5))
        self.mask_ratio = float(mp.get("mi_mask_ratio", 0.15))
        self.use_fomaml = bool(mp.get("mi_use_fomaml", True))

        # Tokenizer pad/eos id'sini model'den almak için trainer'ın
        # set_mask_token_id() ile inject etmesi gerek
        self.mask_token_id: int = 50256  # GPT-2 eos default

        self.outer_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.outer_lr)

    def set_mask_token_id(self, tid: int) -> None:
        self.mask_token_id = tid

    # ─────────────────────────────────────────────────────
    @property
    def requires_meta_loop(self) -> bool:
        return True

    # ─────────────────────────────────────────────────────
    def _self_supervised_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"]
        masked, labels = mask_tokens(
            input_ids, self.mask_token_id, mask_ratio=self.mask_ratio
        )
        out = self.model(input_ids=masked, attention_mask=batch.get("attention_mask"), labels=labels)
        return out.loss

    # ─────────────────────────────────────────────────────
    def adapt(self, context_batch: Dict[str, torch.Tensor]) -> None:
        """Inference-time: küçük sayıda SGD adımı (model üzerinde in-place)."""
        was_training = self.model.training
        self.model.train()
        inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
        for _ in range(self.inner_steps):
            loss = self._self_supervised_loss(context_batch)
            inner_opt.zero_grad()
            loss.backward()
            inner_opt.step()
        if not was_training:
            self.model.eval()

    def meta_train_step(self, support_batch: Dict[str, torch.Tensor], query_batch: Dict[str, torch.Tensor]) -> float:
        """FOMAML: inner SGD → query loss → outer Adam step."""
        original = {n: p.data.clone() for n, p in self.model.named_parameters()}
        self.adapt(support_batch)

        # Outer gradient — adapt edilmiş ağırlıklarda
        query_loss = self._self_supervised_loss(query_batch)
        self.outer_optimizer.zero_grad()
        query_loss.backward()

        # FOMAML: gradyan adapt edilmiş θ'da; outer step orijinal θ üzerinde uygulanır
        for n, p in self.model.named_parameters():
            p.data.copy_(original[n])
        self.outer_optimizer.step()
        return float(query_loss.item())

    def reset(self) -> None:
        # Outer optimizer state'i koru — sadece inner SGD'nin ağırlık değişikliği
        # adapt() içinde otomatik geri alınır (manuel save/restore yok burada
        # çünkü meta_train_step bunu zaten yapıyor; eval'de adapt sonrası
        # reset için snapshot mekaniği trainer'da)
        pass
