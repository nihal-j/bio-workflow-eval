"""
Loads benchmark tasks from tasks/benchmark.json.
Validates them against the BioTask schema on load.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import BioTask

# Default benchmark location relative to the project root
DEFAULT_BENCHMARK = Path(__file__).parents[2] / "tasks" / "benchmark.json"


def load_tasks(path: Path | str | None = None) -> list[BioTask]:
    """Load and validate all tasks from the benchmark file."""
    benchmark_path = Path(path) if path else DEFAULT_BENCHMARK

    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    raw = json.loads(benchmark_path.read_text())
    tasks = [BioTask.model_validate(t) for t in raw]
    return tasks


def load_task_by_id(task_id: str, path: Path | str | None = None) -> BioTask:
    """Load a single task by its task_id."""
    tasks = load_tasks(path)
    for task in tasks:
        if task.task_id == task_id:
            return task
    raise KeyError(f"Task '{task_id}' not found in benchmark")


def task_summary(tasks: list[BioTask]) -> dict:
    """Return a quick summary dict of the loaded task set."""
    from collections import Counter

    return {
        "total": len(tasks),
        "domains": dict(Counter(t.domain for t in tasks)),
        "evidence_statuses": dict(Counter(t.evidence_status for t in tasks)),
        "correct_actions": dict(Counter(t.correct_action for t in tasks)),
    }
