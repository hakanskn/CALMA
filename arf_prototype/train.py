"""Tek run entry point.

Örnekler:
    python train.py --method A1_PCR --seed 42
    python train.py --method baseline --train-ds wikitext2 --eval-ds finance --epochs 1
    python train.py --method A1_PCR --all-seeds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root'u path'e ekle
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src.config import ExperimentConfig, list_methods
from src.training.trainer import run_experiment


def parse_args():
    p = argparse.ArgumentParser(description="ARF prototype — tek run")
    p.add_argument("--method", required=True, choices=list_methods())
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--all-seeds", action="store_true",
                   help="5 seed'i sırayla çalıştır (42,123,456,789,1024)")
    p.add_argument("--train-ds", default=None)
    p.add_argument("--eval-ds", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--limit-train", type=int, default=None,
                   help="Smoke test için kaç batch")
    p.add_argument("--limit-eval", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--drive-base", default=None,
                   help="Override DRIVE_BASE")
    return p.parse_args()


def build_overrides(args, seed: int) -> dict:
    o = {"seed": seed}
    if args.train_ds: o["train_dataset"] = args.train_ds
    if args.eval_ds:  o["eval_dataset"] = args.eval_ds
    if args.epochs is not None: o["num_epochs"] = args.epochs
    if args.batch_size: o["batch_size"] = args.batch_size
    if args.lr is not None: o["learning_rate"] = args.lr
    if args.max_seq_len: o["max_seq_len"] = args.max_seq_len
    if args.limit_train: o["limit_train_batches"] = args.limit_train
    if args.limit_eval: o["limit_eval_batches"] = args.limit_eval
    if args.no_resume: o["resume_from_latest"] = False
    if args.drive_base: o["drive_base"] = args.drive_base
    return o


def main():
    args = parse_args()
    seeds = [42, 123, 456, 789, 1024] if args.all_seeds else [args.seed]

    for s in seeds:
        cfg = ExperimentConfig.load(
            method=args.method,
            overrides=build_overrides(args, s),
        )
        print("=" * 60)
        print(f"▶ {cfg.method} seed={s} run_id={cfg.run_id}")
        print("=" * 60)
        result = run_experiment(cfg, progress_callback=lambda ev: _short_print(ev))
        final = result["final_metrics"]
        print(f"\n✓ Test PPL: {final['test_ppl']:.4f}  BPC: {final['test_bpc']:.4f}")


def _short_print(event: dict) -> None:
    ev = event.get("event")
    if ev == "step":
        print(f"  step={event['step']} loss={event['train_loss']:.4f}")
    elif ev == "epoch_end":
        print(
            f"  epoch={event['epoch']} train={event['train_loss']:.4f} "
            f"val_ppl={event['val_ppl']:.4f}"
        )
    elif ev == "checkpoint":
        print(f"  ckpt @ step {event['step']}")
    elif ev == "model_built":
        print(f"  params total={event['params_total']:,} trainable={event['params_trainable']:,}")
    elif ev in ("start", "resume", "adapt_at_eval", "finished"):
        print(f"  [{ev}] {event}")


if __name__ == "__main__":
    main()
