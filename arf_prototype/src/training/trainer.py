"""Ana eğitim döngüsü.

run_experiment(config) → tam pipeline:
  1. seed
  2. dataset hazırla
  3. model + adapter build
  4. train (epoch loop, log, checkpoint)
  5. val/test evaluate
  6. metrics & config Drive'a yaz
  7. SQLite kayıt
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from ..config import ExperimentConfig
from ..data.dataset_loader import DataModule
from ..models.base_model import build_model
from ..models.learning.base_adapter import build_adapter
from ..utils import (
    count_parameters,
    ensure_dir,
    estimate_remaining_seconds,
    format_duration,
    get_device,
    gpu_memory_mb,
    gpu_name,
    reset_gpu_peak,
    seed_everything,
)
from .checkpointer import Checkpointer
from .evaluator import Evaluator


class Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.config = config
        self.device = get_device()
        seed_everything(config.seed)

        # Data
        self.train_dm = DataModule(
            config.train_dataset,
            tokenizer_name=config.model_name,
            max_seq_len=config.max_seq_len,
            batch_size=config.batch_size,
            limit_train_batches=config.limit_train_batches,
            limit_eval_batches=config.limit_eval_batches,
        )
        if config.eval_dataset == config.train_dataset:
            self.eval_dm = self.train_dm
        else:
            self.eval_dm = DataModule(
                config.eval_dataset,
                tokenizer_name=config.model_name,
                max_seq_len=config.max_seq_len,
                batch_size=config.batch_size,
                limit_eval_batches=config.limit_eval_batches,
            )

        # Model
        self.model = build_model(config).to(self.device)

        # Adapter (Yön B)
        self.adapter = build_adapter(config, self.model)
        if self.adapter and hasattr(self.adapter, "set_mask_token_id"):
            self.adapter.set_mask_token_id(self.train_dm.tokenizer.eos_token_id)

        # Optimizer + scheduler — Yön B'de adapter kendi optimizer'ını kullanır
        if self.adapter and self.adapter.requires_meta_loop:
            self.optimizer = None
            self.scheduler = None
        else:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                trainable, lr=config.learning_rate, weight_decay=config.weight_decay
            )
            self.scheduler = None  # lazy — train_loader hazır olunca kurulacak

        self.checkpointer = Checkpointer(config)
        self.evaluator = Evaluator(self.model, self.device)
        self.progress_callback = progress_callback or (lambda _ev: None)

        # Run state
        self._global_step = 0
        self._start_time = 0.0

    # ─────────────────────────────────────────────────────
    def _build_scheduler(self, num_steps: int) -> None:
        if self.optimizer is None or self.scheduler is not None:
            return
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_steps,
        )

    def _emit(self, event: str, **kwargs) -> None:
        self.progress_callback({"event": event, **kwargs})

    # ─────────────────────────────────────────────────────
    def train(self) -> Dict[str, Any]:
        cfg = self.config
        ensure_dir(cfg.run_dir())
        cfg.save_snapshot()
        reset_gpu_peak()
        self._start_time = time.perf_counter()
        self._emit("start", run_id=cfg.run_id, gpu=gpu_name())

        train_loader = self.train_dm.train_loader()
        val_loader = self.eval_dm.val_loader()

        total_steps = len(train_loader) * cfg.num_epochs
        self._build_scheduler(total_steps)

        # Resume
        if cfg.resume_from_latest:
            start_step = self.checkpointer.load_latest(self.model, self.optimizer)
            if start_step:
                self._emit("resume", from_step=start_step)
                self._global_step = start_step

        n_total = count_parameters(self.model, only_trainable=False)
        n_train = count_parameters(self.model, only_trainable=True)
        self._emit(
            "model_built",
            params_total=n_total,
            params_trainable=n_train,
            method=cfg.method,
        )

        epoch_metrics_log = []
        step_metrics_log = []

        for epoch in range(cfg.num_epochs):
            t_epoch = time.perf_counter()
            train_loss = self._train_one_epoch(train_loader, epoch, step_metrics_log)
            val_metrics = self.evaluator.evaluate(val_loader)
            self.model.train()

            ep_record = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": val_metrics.loss,
                "val_ppl": val_metrics.perplexity,
                "val_bpc": val_metrics.bits_per_char,
                "epoch_time_s": time.perf_counter() - t_epoch,
                "gpu_mem_mb": gpu_memory_mb(),
            }
            epoch_metrics_log.append(ep_record)
            self._emit("epoch_end", **ep_record)

        # Test
        test_loader = self.eval_dm.test_loader()
        if self.adapter and self.adapter.adapts_at_eval:
            self._emit("adapt_at_eval", method=cfg.method)
            self._adapt_on_first_batch(test_loader)
        test_metrics = self.evaluator.evaluate(test_loader)

        # Domain shift δ
        domain_shift_delta = None
        if cfg.train_dataset != cfg.eval_dataset:
            # In-domain reference (train_dataset üzerinde de eval)
            in_loader = self.train_dm.test_loader()
            in_metrics = self.evaluator.evaluate(in_loader)
            domain_shift_delta = test_metrics.perplexity / max(1e-6, in_metrics.perplexity)

        final = {
            "run_id": cfg.run_id,
            "method": cfg.method,
            "seed": cfg.seed,
            "duration_seconds": time.perf_counter() - self._start_time,
            "gpu": gpu_name(),
            "model_params": {
                "total": n_total,
                "trainable": n_train,
            },
            "train_dataset": cfg.train_dataset,
            "eval_dataset": cfg.eval_dataset,
            "epochs": epoch_metrics_log,
            "final_metrics": {
                "test_loss": test_metrics.loss,
                "test_ppl":  test_metrics.perplexity,
                "test_bpc":  test_metrics.bits_per_char,
                "domain_shift_delta": domain_shift_delta,
            },
            "step_metrics": step_metrics_log,
            "config": cfg.to_dict(),
        }
        self._emit("finished", **final["final_metrics"])
        return final

    # ─────────────────────────────────────────────────────
    def _train_one_epoch(
        self, loader: DataLoader, epoch: int, step_log: list
    ) -> float:
        cfg = self.config
        self.model.train()
        running = 0.0
        n_steps = 0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if self.adapter and self.adapter.requires_meta_loop:
                # B1/B2 meta-train step
                from ..data.preprocessor import split_support_query
                support, query = split_support_query(batch)
                loss_val = self.adapter.meta_train_step(support, query)
            else:
                out = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
                loss = out.loss
                self.optimizer.zero_grad()
                loss.backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                loss_val = float(loss.item())

            running += loss_val
            n_steps += 1
            self._global_step += 1

            if self._global_step % cfg.log_every_n_steps == 0:
                step_log.append({
                    "step": self._global_step,
                    "epoch": epoch + 1,
                    "train_loss": loss_val,
                    "eta_seconds": estimate_remaining_seconds(
                        self._global_step,
                        len(loader) * cfg.num_epochs,
                        self._start_time,
                    ),
                })
                self._emit(
                    "step",
                    step=self._global_step,
                    train_loss=loss_val,
                    epoch=epoch + 1,
                )

            if self._global_step % cfg.checkpoint_every_n_steps == 0:
                self.checkpointer.save(
                    self.model, self.optimizer, self._global_step,
                    {"train_loss": loss_val},
                )
                self._emit("checkpoint", step=self._global_step)

        return running / max(1, n_steps)

    def _adapt_on_first_batch(self, loader: DataLoader) -> None:
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.adapter.adapt(batch)
            break


# ─────────────────────────────────────────────────────────
# Tek noktadan giriş — train.py + dashboard ortak kullanır
# ─────────────────────────────────────────────────────────
def run_experiment(
    config: ExperimentConfig,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Tek run'ı baştan sona çalıştır + sonucu logger'a yaz."""
    from ..logging_utils.experiment_logger import ExperimentLogger

    logger = ExperimentLogger(config)
    logger.mark_started()

    def _cb(event):
        logger.handle_event(event)
        if progress_callback:
            progress_callback(event)

    try:
        trainer = Trainer(config, progress_callback=_cb)
        result = trainer.train()
        logger.save_metrics(result)
        logger.mark_completed(result["final_metrics"])
        return result
    except Exception as e:
        logger.mark_failed(str(e))
        raise
