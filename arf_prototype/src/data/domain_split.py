"""Domain shift testleri — train≠eval olduğu özel split üretimi.

Kullanım:
  test = DomainShiftScenario.build(train="wikitext2", eval="finance")
  test.train_loader(), test.eval_loader()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .dataset_loader import DataModule


@dataclass
class DomainShiftScenario:
    train: str
    eval: str
    train_module: DataModule
    eval_module: DataModule

    @classmethod
    def build(
        cls,
        train: str,
        eval: str,
        tokenizer: str = "gpt2",
        max_seq_len: int = 512,
        batch_size: int = 16,
    ) -> "DomainShiftScenario":
        train_dm = DataModule(train, tokenizer, max_seq_len, batch_size)
        eval_dm = DataModule(eval, tokenizer, max_seq_len, batch_size)
        return cls(train=train, eval=eval, train_module=train_dm, eval_module=eval_dm)

    def is_shift(self) -> bool:
        return self.train != self.eval
