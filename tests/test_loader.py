"""Tests for task loading and validation."""

import json
import tempfile
from pathlib import Path

import pytest

from bio_workflow_eval.loader import load_tasks, load_task_by_id, task_summary
from bio_workflow_eval.schemas import BioTask, EvidenceStatus, NextAction


def make_minimal_task(**overrides) -> dict:
    base = {
        "task_id": "test001",
        "domain": "test_domain",
        "question": "Is this a test question?",
        "scenario": "A test scenario.",
        "evidence": ["Evidence snippet one.", "Evidence snippet two."],
        "evidence_status": "sufficient",
        "correct_action": "answer",
        "gold_answer": "Yes, this is a test.",
        "gold_reasoning": "This is clearly a test.",
        "expected_failure_labels": [],
    }
    base.update(overrides)
    return base


def write_tasks_file(tasks: list[dict]) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(tasks, f)
    f.flush()
    return Path(f.name)


class TestLoadTasks:
    def test_loads_real_benchmark(self):
        tasks = load_tasks()
        assert len(tasks) >= 10
        assert all(isinstance(t, BioTask) for t in tasks)

    def test_validates_enum_fields(self):
        tasks = load_tasks()
        for task in tasks:
            assert isinstance(task.evidence_status, EvidenceStatus)
            assert isinstance(task.correct_action, NextAction)

    def test_evidence_count_bounds(self):
        tasks = load_tasks()
        for task in tasks:
            assert 1 <= len(task.evidence) <= 5

    def test_load_from_custom_path(self):
        task_data = [make_minimal_task()]
        path = write_tasks_file(task_data)
        tasks = load_tasks(path)
        assert len(tasks) == 1
        assert tasks[0].task_id == "test001"

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_tasks("/nonexistent/path/benchmark.json")

    def test_raises_on_invalid_schema(self):
        bad_task = make_minimal_task(evidence_status="invalid_value")
        path = write_tasks_file([bad_task])
        with pytest.raises(Exception):
            load_tasks(path)


class TestLoadTaskById:
    def test_finds_existing_task(self):
        tasks = load_tasks()
        first_id = tasks[0].task_id
        task = load_task_by_id(first_id)
        assert task.task_id == first_id

    def test_raises_on_missing_id(self):
        with pytest.raises(KeyError):
            load_task_by_id("does_not_exist_xyz")


class TestTaskSummary:
    def test_summary_fields(self):
        tasks = load_tasks()
        summary = task_summary(tasks)
        assert "total" in summary
        assert "domains" in summary
        assert "evidence_statuses" in summary
        assert "correct_actions" in summary
        assert summary["total"] == len(tasks)

    def test_all_tasks_counted(self):
        tasks = load_tasks()
        summary = task_summary(tasks)
        assert sum(summary["domains"].values()) == summary["total"]
