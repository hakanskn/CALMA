"""Run bazlı loglama — JSON snapshots + SQLite kayıtlar + text log.

Her run için dosyalar:
  - <run_dir>/config.json
  - <run_dir>/metrics.json
  - <run_dir>/log.txt
SQLite DB: <drive_base>/arf_results.db (tüm run'lar tek tabloda)
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils import ensure_dir


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    method          TEXT NOT NULL,
    seed            INTEGER,
    started_at      TIMESTAMP,
    finished_at     TIMESTAMP,
    status          TEXT,
    train_dataset   TEXT,
    eval_dataset    TEXT,
    model_name      TEXT,
    total_params    INTEGER,
    final_test_ppl  REAL,
    final_test_bpc  REAL,
    final_test_loss REAL,
    domain_shift_delta REAL,
    duration_seconds REAL,
    config_json     TEXT
);
CREATE TABLE IF NOT EXISTS epoch_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT REFERENCES runs(run_id),
    epoch           INTEGER,
    train_loss      REAL,
    val_loss        REAL,
    val_ppl         REAL,
    val_bpc         REAL,
    epoch_time_s    REAL,
    gpu_mem_mb      INTEGER
);
CREATE TABLE IF NOT EXISTS step_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT REFERENCES runs(run_id),
    step            INTEGER,
    epoch           INTEGER,
    train_loss      REAL,
    logged_at       TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_runs_method ON runs(method);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
"""


def _db_path(drive_base: str) -> str:
    ensure_dir(drive_base)
    return os.path.join(drive_base, "arf_results.db")


def _connect(drive_base: str) -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(drive_base))
    conn.executescript(SCHEMA)
    return conn


class ExperimentLogger:
    """Tek run için dosya + DB writer."""

    def __init__(self, exp_config):
        self.config = exp_config
        self.run_dir = Path(ensure_dir(exp_config.run_dir()))
        self.log_path = self.run_dir / "log.txt"
        self.metrics_path = self.run_dir / "metrics.json"

    # ─────────────────────────────────────────────────────
    # File operations
    # ─────────────────────────────────────────────────────
    def _append_log(self, line: str) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.utcnow().isoformat()}] {line}\n")

    def save_metrics(self, result: Dict[str, Any]) -> None:
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # ─────────────────────────────────────────────────────
    # DB operations
    # ─────────────────────────────────────────────────────
    def mark_started(self) -> None:
        self._append_log(f"START method={self.config.method} seed={self.config.seed}")
        with _connect(self.config.drive_base) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, method, seed, started_at, status,
                    train_dataset, eval_dataset, model_name, config_json
                ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?)
                """,
                (
                    self.config.run_id,
                    self.config.method,
                    self.config.seed,
                    datetime.utcnow().isoformat(),
                    self.config.train_dataset,
                    self.config.eval_dataset,
                    self.config.model_name,
                    self.config.to_json(),
                ),
            )

    def mark_completed(self, final_metrics: Dict[str, Any]) -> None:
        self._append_log(
            f"COMPLETED test_ppl={final_metrics.get('test_ppl', 0):.4f}"
        )
        with _connect(self.config.drive_base) as conn:
            conn.execute(
                """
                UPDATE runs
                SET finished_at = ?, status = 'completed',
                    final_test_ppl = ?, final_test_bpc = ?, final_test_loss = ?,
                    domain_shift_delta = ?
                WHERE run_id = ?
                """,
                (
                    datetime.utcnow().isoformat(),
                    final_metrics.get("test_ppl"),
                    final_metrics.get("test_bpc"),
                    final_metrics.get("test_loss"),
                    final_metrics.get("domain_shift_delta"),
                    self.config.run_id,
                ),
            )

    def mark_failed(self, error_msg: str) -> None:
        self._append_log(f"FAILED: {error_msg}")
        with _connect(self.config.drive_base) as conn:
            conn.execute(
                "UPDATE runs SET finished_at = ?, status = 'failed' WHERE run_id = ?",
                (datetime.utcnow().isoformat(), self.config.run_id),
            )

    # ─────────────────────────────────────────────────────
    # Event handler — trainer event dispatcher
    # ─────────────────────────────────────────────────────
    def handle_event(self, event: Dict[str, Any]) -> None:
        ev = event.get("event")
        if ev == "epoch_end":
            self._append_log(
                f"EPOCH {event['epoch']} train={event['train_loss']:.4f} "
                f"val_ppl={event['val_ppl']:.4f} mem={event['gpu_mem_mb']:.0f}MB"
            )
            with _connect(self.config.drive_base) as conn:
                conn.execute(
                    """
                    INSERT INTO epoch_metrics
                    (run_id, epoch, train_loss, val_loss, val_ppl, val_bpc, epoch_time_s, gpu_mem_mb)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.config.run_id,
                        event["epoch"],
                        event["train_loss"],
                        event["val_loss"],
                        event["val_ppl"],
                        event["val_bpc"],
                        event["epoch_time_s"],
                        event["gpu_mem_mb"],
                    ),
                )
        elif ev == "step":
            with _connect(self.config.drive_base) as conn:
                conn.execute(
                    """
                    INSERT INTO step_metrics (run_id, step, epoch, train_loss, logged_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        self.config.run_id,
                        event["step"],
                        event.get("epoch"),
                        event["train_loss"],
                        datetime.utcnow().isoformat(),
                    ),
                )
        elif ev == "model_built":
            with _connect(self.config.drive_base) as conn:
                conn.execute(
                    "UPDATE runs SET total_params = ? WHERE run_id = ?",
                    (event["params_total"], self.config.run_id),
                )
            self._append_log(
                f"MODEL params_total={event['params_total']:,} "
                f"trainable={event['params_trainable']:,}"
            )
        elif ev == "checkpoint":
            self._append_log(f"CKPT step={event['step']}")
        elif ev == "start":
            self._append_log(f"GPU: {event.get('gpu', '?')}")
        elif ev == "resume":
            self._append_log(f"RESUME from_step={event['from_step']}")
        elif ev == "adapt_at_eval":
            self._append_log(f"ADAPT_AT_EVAL method={event['method']}")
        elif ev == "finished":
            duration = event.get("duration_seconds")
            with _connect(self.config.drive_base) as conn:
                conn.execute(
                    "UPDATE runs SET duration_seconds = ? WHERE run_id = ?",
                    (duration, self.config.run_id),
                )
