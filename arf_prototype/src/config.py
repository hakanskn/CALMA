"""ExperimentConfig dataclass — tüm runtime parametreler buradan akar.

YAML merge mantığı:
1) base_config.yaml her zaman yüklenir
2) method config (örn. A1_pcr.yaml) base'i override eder
3) CLI / dashboard parametreleri en son override eder
"""

from __future__ import annotations

import os
import yaml
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


@dataclass
class ExperimentConfig:
    # ── Run kimliği ─────────────────────────────────────
    run_id: str = field(default_factory=lambda: f"run_{_utc_stamp()}")
    method: str = "baseline"
    seed: int = 42

    # ── Model ───────────────────────────────────────────
    model_name: str = "gpt2"
    model_hidden_size: int = 768
    model_num_heads: int = 12
    model_num_layers: int = 12

    # ── Eğitim ──────────────────────────────────────────
    learning_rate: float = 5e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 100
    grad_clip: float = 1.0
    weight_decay: float = 0.01

    # ── Seed yönetimi ───────────────────────────────────
    num_seeds: int = 5
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])

    # ── Checkpoint ──────────────────────────────────────
    checkpoint_every_n_steps: int = 500
    keep_last_n_checkpoints: int = 3
    resume_from_latest: bool = True

    # ── Kayıt ───────────────────────────────────────────
    log_every_n_steps: int = 50
    drive_base: str = "/content/drive/MyDrive/arf_results"

    # ── Dataset ─────────────────────────────────────────
    train_dataset: str = "wikitext2"
    eval_dataset: str = "wikitext2"
    max_seq_len: int = 512
    limit_train_batches: Optional[int] = None
    limit_eval_batches: Optional[int] = None

    # ── Method-specific parametreler ────────────────────
    method_params: Dict[str, Any] = field(default_factory=dict)

    # ── Meta ────────────────────────────────────────────
    description: str = ""
    expected_extra_params: int = 0
    notes: str = ""

    # ─────────────────────────────────────────────────────
    # I/O helpers
    # ─────────────────────────────────────────────────────
    def run_dir(self) -> str:
        return os.path.join(self.drive_base, "runs", self.run_id)

    def checkpoint_dir(self) -> str:
        return os.path.join(self.run_dir(), "checkpoints")

    def plots_dir(self) -> str:
        return os.path.join(self.run_dir(), "plots")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def save_snapshot(self) -> str:
        """run_dir'e config.json snapshot'ı yaz."""
        os.makedirs(self.run_dir(), exist_ok=True)
        path = os.path.join(self.run_dir(), "config.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        return path

    # ─────────────────────────────────────────────────────
    # Yükleyiciler
    # ─────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        # Bilinmeyen alanları method_params'a değil; reddet
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    @classmethod
    def load(
        cls,
        method: str = "baseline",
        overrides: Optional[Dict[str, Any]] = None,
        base_path: Optional[str | Path] = None,
        method_path: Optional[str | Path] = None,
    ) -> "ExperimentConfig":
        """Base + method + override zinciri."""
        base_path = Path(base_path) if base_path else CONFIGS_DIR / "base_config.yaml"
        with open(base_path, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}

        if method_path is None:
            method_path = CONFIGS_DIR / "methods" / f"{_method_filename(method)}.yaml"
        with open(method_path, "r", encoding="utf-8") as f:
            method_cfg = yaml.safe_load(f) or {}

        merged = _deep_merge(base, method_cfg)
        if overrides:
            merged = _deep_merge(merged, overrides)

        return cls.from_dict(merged)


def _method_filename(method: str) -> str:
    """`A1_PCR` -> `A1_pcr`, `baseline` -> `baseline`. Dosya adı kuralı."""
    if method == "baseline":
        return "baseline"
    if "_" in method:
        prefix, rest = method.split("_", 1)
        return f"{prefix}_{rest.lower()}"
    return method.lower()


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """b -> a override; dict alanlarında recursive."""
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# 9 yöntem + baseline registry — dashboard ve batch runner için
METHOD_REGISTRY: Dict[str, str] = {
    "baseline":  "Standart Scaled Dot-Product Attention",
    "A1_PCR":    "Predictive Coding Router",
    "A2_LI_SCR": "Lateral Inhibition Sparse Router",
    "A3_RBR":    "Resonance-Based Router",
    "B1_MI_S3T": "Meta-Initialized Self-Supervised TTT",
    "B2_CMA":    "Contrastive Meta-Adaptation",
    "B3_FWML":   "Fast Weight Meta-Learning",
    "C1_PCUN":   "Predictive Coding Unified Network",
    "C2_EP_T":   "Equilibrium Propagation Transformer",
    "C3_CHA":    "Contrastive Hebbian Attention",
}

DIRECTION_GROUPS: Dict[str, List[str]] = {
    "A": ["A1_PCR", "A2_LI_SCR", "A3_RBR"],
    "B": ["B1_MI_S3T", "B2_CMA", "B3_FWML"],
    "C": ["C1_PCUN", "C2_EP_T", "C3_CHA"],
    "ALL": ["baseline"] + [
        "A1_PCR", "A2_LI_SCR", "A3_RBR",
        "B1_MI_S3T", "B2_CMA", "B3_FWML",
        "C1_PCUN", "C2_EP_T", "C3_CHA",
    ],
}


def list_methods() -> List[str]:
    return list(METHOD_REGISTRY.keys())


def list_datasets() -> List[str]:
    ds_dir = CONFIGS_DIR / "datasets"
    return [p.stem for p in ds_dir.glob("*.yaml") if p.stem != "domain_shift"] + [
        "finance", "medical", "legal", "science", "news", "code",
    ]
