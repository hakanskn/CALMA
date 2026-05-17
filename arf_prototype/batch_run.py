"""Toplu run entry point — yön grubu × seed kombinasyonları.

Örnekler:
    python batch_run.py --direction A
    python batch_run.py --all --all-seeds
    python batch_run.py --methods baseline,A1_PCR --seeds 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src.config import DIRECTION_GROUPS, ExperimentConfig, list_methods
from src.training.trainer import run_experiment


def parse_args():
    p = argparse.ArgumentParser(description="ARF prototype — batch run")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--direction", choices=["A", "B", "C", "ALL"], default=None)
    g.add_argument("--all", action="store_true", help="ALL grubu")
    g.add_argument("--methods", default=None,
                   help="Virgülle yöntem listesi, örn: baseline,A1_PCR")

    p.add_argument("--seeds", type=int, default=1,
                   help="Kaç seed (1-5)")
    p.add_argument("--all-seeds", action="store_true")

    p.add_argument("--train-ds", default=None)
    p.add_argument("--eval-ds", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-eval", type=int, default=None)
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    elif args.all:
        methods = DIRECTION_GROUPS["ALL"]
    elif args.direction:
        methods = DIRECTION_GROUPS[args.direction]
    else:
        print("--direction, --all veya --methods belirt.", file=sys.stderr)
        sys.exit(2)

    seeds = [42, 123, 456, 789, 1024]
    seeds = seeds if args.all_seeds else seeds[: max(1, args.seeds)]

    print(f"Methods: {methods}\nSeeds: {seeds}\nTotal runs: {len(methods)*len(seeds)}")

    overrides_common = {}
    if args.train_ds: overrides_common["train_dataset"] = args.train_ds
    if args.eval_ds:  overrides_common["eval_dataset"]  = args.eval_ds
    if args.epochs is not None: overrides_common["num_epochs"] = args.epochs
    if args.batch_size: overrides_common["batch_size"] = args.batch_size
    if args.limit_train: overrides_common["limit_train_batches"] = args.limit_train
    if args.limit_eval: overrides_common["limit_eval_batches"] = args.limit_eval

    results = []
    for m in methods:
        for s in seeds:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            o = dict(overrides_common, seed=s, run_id=f"run_{m}_s{s}_{stamp}")
            cfg = ExperimentConfig.load(method=m, overrides=o)
            print(f"\n▶ {m} seed={s}  →  {cfg.run_id}")
            try:
                r = run_experiment(cfg)
                ppl = r["final_metrics"]["test_ppl"]
                results.append((m, s, ppl, "ok"))
                print(f"  ✓ PPL={ppl:.4f}")
            except Exception as e:
                results.append((m, s, None, f"err: {e}"))
                print(f"  ✗ FAILED: {e}")
                if not args.continue_on_error:
                    raise

    print("\n══ Summary ══")
    for m, s, ppl, st in results:
        ppl_str = f"{ppl:.4f}" if ppl is not None else "—"
        print(f"  {m:<12} seed={s}  PPL={ppl_str}  [{st}]")


if __name__ == "__main__":
    main()
