"""Tokenization yardımcıları — BERT-style masking (B1, B2 için), batch helpers."""

from __future__ import annotations

from typing import Tuple

import torch


def mask_tokens(
    input_ids: torch.Tensor,
    mask_token_id: int,
    mask_ratio: float = 0.15,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """BERT-style masking for masked-LM objective on GPT-2 tokenizer.

    Returns: (masked_input_ids, labels)  — labels[~masked] = ignore_index
    """
    input_ids = input_ids.clone()
    labels = input_ids.clone()
    probability = torch.full(input_ids.shape, mask_ratio, device=input_ids.device)
    masked_indices = torch.bernoulli(probability).bool()
    labels[~masked_indices] = ignore_index
    input_ids[masked_indices] = mask_token_id
    return input_ids, labels


def split_support_query(batch: dict, ratio: float = 0.5) -> Tuple[dict, dict]:
    """Bir batch'i support/query alt-batchlerine ayır — MAML için."""
    bs = batch["input_ids"].size(0)
    cut = max(1, int(bs * ratio))
    support = {k: v[:cut] for k, v in batch.items()}
    query = {k: v[cut:] for k, v in batch.items()}
    return support, query
