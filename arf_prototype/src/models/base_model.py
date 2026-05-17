"""GPT-2 wrapper — yöntemleri attention katmanına inject etmek için tek arayüz.

Tek noktadan model değiştirme: `config.model_name = 'gpt2-medium'` yeterli.
Yön A yöntemleri her transformer bloğunun attention modülünü değiştirir.
Yön B yöntemleri model mimarisini değiştirmez, sadece trainer'a katman ekler.
Yön C yöntemleri ya attention'ı değiştirir ya da forward'ı sarar.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

SUPPORTED_MODELS = {
    "gpt2":        117_000_000,
    "gpt2-medium": 345_000_000,
    "gpt2-large":  774_000_000,
}


class ARFBaseModel(nn.Module):
    """GPT-2 LM head wrapper.

    inject_attention: 9 yöntemin attention modülünü tüm katmanlara uygular.
    Standart kullanım için herhangi bir injection gerekmez — pure GPT-2.
    """

    def __init__(self, exp_config):
        super().__init__()
        self.exp_config = exp_config
        if exp_config.model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Desteklenmeyen model: {exp_config.model_name}. "
                f"Seçenekler: {list(SUPPORTED_MODELS)}"
            )
        self.gpt2_config = GPT2Config.from_pretrained(exp_config.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(exp_config.model_name)
        # Tokenizer pad'i için resize gerekmez — GPT-2 EOS'u pad olarak kullanır

        self.replaced_attention_class: Optional[type] = None
        self.attention_modules: list[nn.Module] = []

    # ─────────────────────────────────────────────────────
    # Attention injection — Yön A & Yön C için
    # ─────────────────────────────────────────────────────
    def inject_attention(self, attention_cls: Callable[..., nn.Module]) -> None:
        """Her transformer bloğunun .attn modülünü yenisiyle değiştir."""
        self.replaced_attention_class = attention_cls
        self.attention_modules = []
        for block in self.model.transformer.h:
            new_attn = attention_cls(self.exp_config, self.gpt2_config)
            block.attn = new_attn
            self.attention_modules.append(new_attn)

    # ─────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def num_parameters(self, only_trainable: bool = True) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def build_model(exp_config) -> ARFBaseModel:
    """ExperimentConfig'ten model üret + uygun yöntemi inject et.

    Yön A & bazı Yön C → inject_attention
    Yön B → mimari değişmez, model olduğu gibi döner
    """
    model = ARFBaseModel(exp_config)

    method = exp_config.method
    if method == "baseline":
        return model

    # Yön A
    if method == "A1_PCR":
        from .attention.A1_pcr import PredictiveCodingRouter
        model.inject_attention(PredictiveCodingRouter)
    elif method == "A2_LI_SCR":
        from .attention.A2_li_scr import LateralInhibitionRouter
        model.inject_attention(LateralInhibitionRouter)
    elif method == "A3_RBR":
        from .attention.A3_rbr import ResonanceBasedRouter
        model.inject_attention(ResonanceBasedRouter)

    # Yön B (sadece trainer'a etki eder — mimari değişmez)
    elif method in {"B1_MI_S3T", "B2_CMA", "B3_FWML"}:
        return model

    # Yön C
    elif method == "C1_PCUN":
        from .unified.C1_pcun import PCUnifiedAttention
        model.inject_attention(PCUnifiedAttention)
    elif method == "C2_EP_T":
        from .unified.C2_ep_t import EquilibriumPropAttention
        model.inject_attention(EquilibriumPropAttention)
    elif method == "C3_CHA":
        from .unified.C3_cha import ContrastiveHebbianAttention
        model.inject_attention(ContrastiveHebbianAttention)
    else:
        raise ValueError(f"Bilinmeyen method: {method}")

    return model
