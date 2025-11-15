from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import DEFAULT_DB_PATH


class Storage:
    """
    Thin wrapper around SQLite for:
    - experiments
    - runs
    - trials

    This is deliberately simple and synchronous for the MVP.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        # `check_same_thread=False` to keep it simple if we later use threads.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        cur = self._conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                description TEXT,
                config_json TEXT,
                created_at TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                name TEXT,
                status TEXT,
                dataset_size INTEGER,
                summary_metrics TEXT,
                created_at TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                example_index INTEGER,
                model_name TEXT,
                prompt_template TEXT,
                input_text TEXT,
                expected_output TEXT,
                output_text TEXT,
                metrics_json TEXT,
                latency_ms REAL,
                created_at TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )
            """
        )

        self._conn.commit()

    # ---------- Experiments ----------

    def get_or_create_experiment(
        self, name: str, description: str, config: Dict[str, Any]
    ) -> int:
        """Return experiment id; create experiment if not exists."""
        cur = self._conn.cursor()
        cur.execute("SELECT id FROM experiments WHERE name = ?", (name,))
        row = cur.fetchone()
        if row:
            return int(row["id"])

        created_at = datetime.utcnow().isoformat()
        cur.execute(
            """
            INSERT INTO experiments (name, description, config_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (name, description, json.dumps(config), created_at),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    # ---------- Runs ----------

    def create_run(
        self,
        experiment_id: int,
        name: str,
        dataset_size: int,
    ) -> int:
        """Create a run row with status 'running'."""
        cur = self._conn.cursor()
        created_at = datetime.utcnow().isoformat()
        cur.execute(
            """
            INSERT INTO runs (
                experiment_id, name, status, dataset_size, summary_metrics, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (experiment_id, name, "running", dataset_size, json.dumps({}), created_at),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def update_run_summary(
        self, run_id: int, status: str, summary_metrics: Dict[str, Any]
    ) -> None:
        """Update run status and summary metrics JSON."""
        cur = self._conn.cursor()
        cur.execute(
            """
            UPDATE runs
            SET status = ?, summary_metrics = ?
            WHERE id = ?
            """,
            (status, json.dumps(summary_metrics), run_id),
        )
        self._conn.commit()

    # ---------- Trials ----------

    def log_trial(
        self,
        run_id: int,
        example_index: int,
        model_name: str,
        prompt_template: str,
        input_text: str,
        expected_output: str,
        output_text: str,
        metrics: Dict[str, Any],
        latency_ms: float,
    ) -> None:
        """Insert a single trial row."""
        cur = self._conn.cursor()
        created_at = datetime.utcnow().isoformat()
        cur.execute(
            """
            INSERT INTO trials (
                run_id, example_index, model_name, prompt_template,
                input_text, expected_output, output_text,
                metrics_json, latency_ms, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                example_index,
                model_name,
                prompt_template,
                input_text,
                expected_output,
                output_text,
                json.dumps(metrics),
                float(latency_ms),
                created_at,
            ),
        )
        self._conn.commit()

    # ---------- Queries for analysis ----------

    def fetch_runs(self) -> List[sqlite3.Row]:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT runs.*, experiments.name AS experiment_name
            FROM runs
            JOIN experiments ON runs.experiment_id = experiments.id
            ORDER BY runs.created_at DESC
            """
        )
        return cur.fetchall()

    def fetch_run(self, run_id: int) -> Optional[sqlite3.Row]:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT runs.*, experiments.name AS experiment_name
            FROM runs
            JOIN experiments ON runs.experiment_id = experiments.id
            WHERE runs.id = ?
            """,
            (run_id,),
        )
        return cur.fetchone()

    def fetch_trials(self, run_id: int) -> List[sqlite3.Row]:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM trials
            WHERE run_id = ?
            ORDER BY example_index ASC
            """,
            (run_id,),
        )
        return cur.fetchall()

    def close(self) -> None:
        self._conn.close()
