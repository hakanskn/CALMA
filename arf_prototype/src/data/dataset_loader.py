"""HuggingFace tabanlı veri yükleme + train/val/test split + GPT-2 tokenizer.

Tek arayüz: `DataModule(dataset_name, tokenizer_name, max_seq_len, batch_size)`.

Domain shift için `train_dataset != eval_dataset` desteklenir.
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer

from ..config import CONFIGS_DIR


DOMAIN_SHIFT_KEYS = {"finance", "medical", "legal", "science", "news", "code"}


def load_dataset_config(name: str) -> Dict[str, Any]:
    """`wikitext2.yaml` veya `domain_shift.yaml::finance` mantığı."""
    if name in DOMAIN_SHIFT_KEYS:
        ds_path = CONFIGS_DIR / "datasets" / "domain_shift.yaml"
        with open(ds_path, "r", encoding="utf-8") as f:
            domain_map = yaml.safe_load(f)
        if name not in domain_map:
            raise KeyError(f"Domain config bulunamadı: {name}")
        return domain_map[name]

    ds_path = CONFIGS_DIR / "datasets" / f"{name}.yaml"
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset config yok: {ds_path}")
    with open(ds_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TokenizedTextDataset(Dataset):
    """Tokenize edilmiş satırların düz listesi. GPT-2 causal LM için."""

    def __init__(self, hf_ds: HFDataset, tokenizer, text_column: str, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        # eos token'ı tokenizer'da yoksa pad olarak ekle
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tüm metni concat et, max_len bloklara böl — Karpathy/GPT-2 stili
        texts = [t for t in hf_ds[text_column] if isinstance(t, str) and t.strip()]
        joined = "\n\n".join(texts)
        tokens = self.tokenizer(joined, return_tensors="pt").input_ids[0]
        n_blocks = tokens.size(0) // max_len
        if n_blocks == 0:
            self.blocks = tokens.unsqueeze(0)[:, :max_len]
        else:
            self.blocks = tokens[: n_blocks * max_len].view(n_blocks, max_len)

    def __len__(self) -> int:
        return self.blocks.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.blocks[idx]
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
        }


def _hf_load(cfg: Dict[str, Any]) -> Dict[str, HFDataset]:
    """HuggingFace dataset'ini split bazlı yükle. Bazı setlerde train/test/val olur."""
    kwargs = {}
    if cfg.get("hf_config"):
        kwargs["name"] = cfg["hf_config"]
    if cfg.get("streaming"):
        kwargs["streaming"] = True
    raw = load_dataset(cfg["hf_id"], **kwargs)

    if isinstance(raw, dict):
        return raw
    # streaming veya tek split
    return {"train": raw}


def _build_splits(
    raw_splits: Dict[str, HFDataset],
    split_ratios: Dict[str, float],
) -> Tuple[HFDataset, HFDataset, HFDataset]:
    """Verilen HF dataset'i train/val/test'e böl."""
    # Eğer zaten train/validation/test varsa onları kullan
    has_train = "train" in raw_splits
    has_val = any(k in raw_splits for k in ("validation", "val", "dev"))
    has_test = "test" in raw_splits

    if has_train and has_val and has_test:
        train = raw_splits["train"]
        val = raw_splits.get("validation") or raw_splits.get("val") or raw_splits.get("dev")
        test = raw_splits["test"]
        return train, val, test

    src = raw_splits.get("train") or list(raw_splits.values())[0]
    n = len(src)
    n_train = int(n * split_ratios.get("train", 0.8))
    n_val = int(n * split_ratios.get("val", 0.1))
    indices = list(range(n))
    return (
        src.select(indices[:n_train]),
        src.select(indices[n_train : n_train + n_val]),
        src.select(indices[n_train + n_val :]),
    )


class DataModule:
    """Eğitim + değerlendirme dataloader'larını tek noktada toplar."""

    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str = "gpt2",
        max_seq_len: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_workers: int = 2,
        limit_train_batches: Optional[int] = None,
        limit_eval_batches: Optional[int] = None,
    ):
        self.cfg = load_dataset_config(dataset_name)
        self.dataset_name = dataset_name
        self.max_seq_len = max_seq_len or self.cfg.get("max_seq_len", 512)
        self.batch_size = batch_size or self.cfg.get("batch_size", 16)
        self.num_workers = num_workers
        self.limit_train_batches = limit_train_batches
        self.limit_eval_batches = limit_eval_batches

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._train_ds: Optional[TokenizedTextDataset] = None
        self._val_ds: Optional[TokenizedTextDataset] = None
        self._test_ds: Optional[TokenizedTextDataset] = None

    def prepare(self) -> None:
        if self._train_ds is not None:
            return
        raw = _hf_load(self.cfg)
        train, val, test = _build_splits(raw, self.cfg.get("split_ratios", {}))
        text_col = self.cfg.get("text_column", "text")
        self._train_ds = TokenizedTextDataset(train, self.tokenizer, text_col, self.max_seq_len)
        self._val_ds = TokenizedTextDataset(val, self.tokenizer, text_col, self.max_seq_len)
        self._test_ds = TokenizedTextDataset(test, self.tokenizer, text_col, self.max_seq_len)

    def _maybe_limit(self, ds, n: Optional[int]):
        if n is None or n <= 0 or len(ds) <= n * self.batch_size:
            return ds
        from torch.utils.data import Subset
        return Subset(ds, list(range(n * self.batch_size)))

    def train_loader(self) -> DataLoader:
        self.prepare()
        ds = self._maybe_limit(self._train_ds, self.limit_train_batches)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    def val_loader(self) -> DataLoader:
        self.prepare()
        ds = self._maybe_limit(self._val_ds, self.limit_eval_batches)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_loader(self) -> DataLoader:
        self.prepare()
        ds = self._maybe_limit(self._test_ds, self.limit_eval_batches)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # Convenience
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        return self.train_loader(), self.val_loader(), self.test_loader()
