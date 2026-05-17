"""Read-only erişim: dashboard ve karşılaştırma scriptleri için.

ResultStore tek bir DB path'i alır, query yardımcıları sağlar.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class RunSummary:
    run_id: str
    method: str
    seed: int
    status: str
    test_ppl: Optional[float]
    test_bpc: Optional[float]
    duration_seconds: Optional[float]
    train_dataset: str
    eval_dataset: str
    started_at: str
    finished_at: Optional[str]
    total_params: Optional[int]
    domain_shift_delta: Optional[float]


class ResultStore:
    def __init__(self, drive_base: str):
        self.drive_base = drive_base
        self.db_path = os.path.join(drive_base, "arf_results.db")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ─────────────────────────────────────────────────────
    def list_runs(
        self,
        status: Optional[str] = None,
        method: Optional[str] = None,
        train_dataset: Optional[str] = None,
        eval_dataset: Optional[str] = None,
    ) -> List[RunSummary]:
        sql = "SELECT * FROM runs WHERE 1=1"
        params: List[Any] = []
        if status:
            sql += " AND status = ?"
            params.append(status)
        if method:
            sql += " AND method = ?"
            params.append(method)
        if train_dataset:
            sql += " AND train_dataset = ?"
            params.append(train_dataset)
        if eval_dataset:
            sql += " AND eval_dataset = ?"
            params.append(eval_dataset)
        sql += " ORDER BY started_at DESC"
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            RunSummary(
                run_id=r["run_id"],
                method=r["method"],
                seed=r["seed"],
                status=r["status"],
                test_ppl=r["final_test_ppl"],
                test_bpc=r["final_test_bpc"],
                duration_seconds=r["duration_seconds"],
                train_dataset=r["train_dataset"],
                eval_dataset=r["eval_dataset"],
                started_at=r["started_at"],
                finished_at=r["finished_at"],
                total_params=r["total_params"],
                domain_shift_delta=r["domain_shift_delta"],
            )
            for r in rows
        ]

    def runs_df(self, **filters) -> pd.DataFrame:
        runs = self.list_runs(**filters)
        if not runs:
            return pd.DataFrame()
        return pd.DataFrame([r.__dict__ for r in runs])

    def epoch_metrics_df(self, run_id: str) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT * FROM epoch_metrics WHERE run_id = ? ORDER BY epoch",
                conn,
                params=(run_id,),
            )

    def step_metrics_df(self, run_id: str) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT * FROM step_metrics WHERE run_id = ? ORDER BY step",
                conn,
                params=(run_id,),
            )

    def compare_methods(self, methods: List[str]) -> pd.DataFrame:
        """Her method için 5-seed ortalama/sapma."""
        if not methods:
            return pd.DataFrame()
        placeholders = ",".join("?" * len(methods))
        sql = f"""
        SELECT method,
               AVG(final_test_ppl) AS ppl_mean,
               COUNT(*)            AS n_runs,
               MIN(final_test_ppl) AS ppl_min,
               MAX(final_test_ppl) AS ppl_max,
               AVG(total_params)   AS params_mean,
               AVG(duration_seconds) AS duration_mean
        FROM runs
        WHERE status = 'completed' AND method IN ({placeholders})
        GROUP BY method
        ORDER BY ppl_mean
        """
        with self._conn() as conn:
            df = pd.read_sql_query(sql, conn, params=methods)
        # Standart sapma manuel — SQLite STDEV yok
        std_rows = []
        for m in df["method"]:
            with self._conn() as conn:
                vals = conn.execute(
                    "SELECT final_test_ppl FROM runs WHERE method=? AND status='completed'",
                    (m,),
                ).fetchall()
            arr = [v[0] for v in vals if v[0] is not None]
            if len(arr) > 1:
                import statistics
                std_rows.append(statistics.stdev(arr))
            else:
                std_rows.append(0.0)
        df["ppl_std"] = std_rows
        return df

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def delete_run(self, run_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM step_metrics WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM epoch_metrics WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
