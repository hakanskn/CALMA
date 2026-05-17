"""Yön B — Inference-time adaptation yöntemleri.

Trainer'a entegrasyon noktaları:
  - meta_train_step(support, query) : outer loop (Faz 2'de aktif)
  - adapt(context_batch)            : inference-time iç döngü
  - reset()                          : oturum sonu sıfırla
"""

from .base_adapter import BaseAdaptationMethod

__all__ = ["BaseAdaptationMethod"]
