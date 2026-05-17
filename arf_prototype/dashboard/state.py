"""Dashboard ile arka plan training thread'i arasındaki köprü.

Tek bir aktif run olabilir — Colab GPU'sunu paylaşıyoruz. Batch run sıralıdır.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.config import ExperimentConfig
from src.training.trainer import run_experiment


@dataclass
class RunHandle:
    run_id: str
    method: str
    seed: int
    status: str = "pending"          # pending | running | completed | failed
    started_at: float = 0.0
    ended_at: Optional[float] = None
    error: Optional[str] = None
    last_event: Dict[str, Any] = field(default_factory=dict)
    log_lines: List[str] = field(default_factory=list)


class DashboardState:
    """Singleton-ish — Gradio app yaşam döngüsünde bir tane."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self.current: Optional[RunHandle] = None
        self.queue: List[ExperimentConfig] = []          # bekleyen batch
        self.history: List[RunHandle] = []
        self.event_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=10000)

    # ─────────────────────────────────────────────────────
    def is_running(self) -> bool:
        return self.current is not None and self.current.status == "running"

    def enqueue(self, configs: List[ExperimentConfig]) -> None:
        with self._lock:
            self.queue.extend(configs)
        self._maybe_start_worker()

    def submit_single(self, cfg: ExperimentConfig) -> str:
        with self._lock:
            self.queue.append(cfg)
        self._maybe_start_worker()
        return cfg.run_id

    def stop(self) -> None:
        self._stop_flag.set()

    def reset_stop(self) -> None:
        self._stop_flag = threading.Event()

    # ─────────────────────────────────────────────────────
    def _maybe_start_worker(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.reset_stop()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _emit(self, line: str) -> None:
        if self.current is not None:
            self.current.log_lines.append(line)
            if len(self.current.log_lines) > 500:
                self.current.log_lines = self.current.log_lines[-500:]

    def _callback(self, event: Dict[str, Any]) -> None:
        ev = event.get("event", "")
        if self.current is not None:
            self.current.last_event = event
        # Compact log line
        msg = f"[{ev}]"
        for k in ("epoch", "step", "train_loss", "val_ppl", "val_loss", "gpu", "params_total"):
            if k in event:
                v = event[k]
                if isinstance(v, float):
                    msg += f" {k}={v:.4f}"
                else:
                    msg += f" {k}={v}"
        self._emit(msg)
        try:
            self.event_q.put_nowait(event)
        except queue.Full:
            pass

    def _worker(self) -> None:
        while True:
            with self._lock:
                if self._stop_flag.is_set():
                    return
                if not self.queue:
                    return
                cfg = self.queue.pop(0)

            handle = RunHandle(
                run_id=cfg.run_id, method=cfg.method, seed=cfg.seed,
                status="running", started_at=time.time(),
            )
            self.current = handle
            self.history.append(handle)
            try:
                run_experiment(cfg, progress_callback=self._callback)
                handle.status = "completed"
            except Exception as e:
                handle.status = "failed"
                handle.error = f"{e}\n{traceback.format_exc()}"
                self._emit(f"[error] {e}")
            finally:
                handle.ended_at = time.time()
                self.current = None

    # ─────────────────────────────────────────────────────
    def snapshot(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running(),
            "queue_size": len(self.queue),
            "current": (
                {
                    "run_id":  self.current.run_id,
                    "method":  self.current.method,
                    "seed":    self.current.seed,
                    "status":  self.current.status,
                    "log_tail": "\n".join(self.current.log_lines[-25:]),
                    "last_event": self.current.last_event,
                }
                if self.current
                else None
            ),
            "history": [
                {
                    "run_id": h.run_id,
                    "method": h.method,
                    "seed": h.seed,
                    "status": h.status,
                    "error": (h.error or "")[:200],
                }
                for h in self.history[-30:]
            ],
        }


# Module-level singleton
state = DashboardState()
