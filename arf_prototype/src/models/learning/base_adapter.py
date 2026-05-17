"""Yön B için ortak adapter arayüzü.

Trainer içinde şu noktalarda çağrılır:
  - train batch'i ile:                meta_train_step()  (opsiyonel)
  - eval başında, her context için:   adapt(batch)
  - run/episode sonu:                  reset()
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class BaseAdaptationMethod:
    """Abstract base — concrete subclass'lar method-specific davranışı ekler."""

    def __init__(self, model: nn.Module, exp_config):
        self.model = model
        self.exp_config = exp_config
        self.method_params = exp_config.method_params

    # ─────────────────────────────────────────────────────
    # Trainer entegrasyon noktaları
    # ─────────────────────────────────────────────────────
    def meta_train_step(self, support_batch: dict, query_batch: dict) -> float:
        """Outer loop bir adım — query loss değerini döner."""
        raise NotImplementedError

    def adapt(self, context_batch: dict) -> None:
        """Inference-time inner loop — ağırlıkları context'e uydur."""
        raise NotImplementedError

    def reset(self) -> None:
        """Adapte olmuş ağırlıkları başlangıca al."""
        raise NotImplementedError

    # ─────────────────────────────────────────────────────
    # Trainer için flag'ler
    # ─────────────────────────────────────────────────────
    @property
    def adapts_at_eval(self) -> bool:
        return bool(self.method_params.get("adapt_at_eval", False))

    @property
    def requires_meta_loop(self) -> bool:
        """Eğitim sırasında MAML/CMA tarzı outer loop çalıştırılmalı mı?"""
        return False


def build_adapter(exp_config, model: nn.Module) -> Optional[BaseAdaptationMethod]:
    """Method ismine göre uygun Yön B adapter'ını döner; baseline/A/C için None."""
    method = exp_config.method
    if method == "B1_MI_S3T":
        from .B1_mi_s3t import MISSelfSupervisedTTT
        return MISSelfSupervisedTTT(model, exp_config)
    if method == "B2_CMA":
        from .B2_cma import ContrastiveMetaAdaptation
        return ContrastiveMetaAdaptation(model, exp_config)
    if method == "B3_FWML":
        from .B3_fwml import FastWeightMetaLearning
        return FastWeightMetaLearning(model, exp_config)
    return None
