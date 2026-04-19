"""Tests for runner and failure detection."""

import json
import tempfile
from pathlib import Path

import pytest

from bio_workflow_eval.schemas import (
    BioTask,
    EvidenceStatus,
    FailureLabel,
    ModelOutput,
    NextAction,
)
from bio_workflow_eval.failures import detect_failures
from bio_workflow_eval.runner import run_evaluation


def make_task(**overrides) -> BioTask:
    base = dict(
        task_id="t001",
        domain="test",
        question="Is X true?",
        scenario="Context.",
        evidence=["Snippet A.", "However, snippet B contradicts A."],
        evidence_status=EvidenceStatus.sufficient,
        correct_action=NextAction.answer,
        gold_answer="X is true.",
        gold_reasoning="Snippet A supports it.",
        expected_failure_labels=[],
    )
    base.update(overrides)
    return BioTask(**base)


def make_output(**overrides) -> ModelOutput:
    base = dict(
        task_id="t001",
        predicted_action=NextAction.answer,
        answer_text="X is true.",
        reasoning_trace="Based on snippet A.",
        cited_evidence_indices=[0],
    )
    base.update(overrides)
    return ModelOutput(**base)


class TestDetectFailures:
    def test_no_failures_on_correct_answer(self):
        task = make_task(evidence_status=EvidenceStatus.sufficient)
        output = make_output(cited_evidence_indices=[0, 1])
        failures = detect_failures(task, output)
        assert FailureLabel.answered_too_early not in failures
        assert FailureLabel.ignored_conflicting_evidence not in failures

    def test_answered_too_early_on_insufficient(self):
        task = make_task(evidence_status=EvidenceStatus.insufficient)
        output = make_output(predicted_action=NextAction.answer)
        failures = detect_failures(task, output)
        assert FailureLabel.answered_too_early in failures

    def test_ignored_conflict_on_conflicting_answer(self):
        task = make_task(evidence_status=EvidenceStatus.conflicting)
        output = make_output(predicted_action=NextAction.answer)
        failures = detect_failures(task, output)
        assert FailureLabel.ignored_conflicting_evidence in failures

    def test_shallow_evidence_when_citing_too_few(self):
        task = make_task(
            evidence=["A.", "B.", "C.", "D."],
            evidence_status=EvidenceStatus.sufficient,
        )
        output = make_output(cited_evidence_indices=[0])  # only 1 of 4
        failures = detect_failures(task, output)
        assert FailureLabel.shallow_evidence_use in failures

    def test_unsupported_claim_with_claim_language_no_citation(self):
        task = make_task()
        output = make_output(
            answer_text="This confirms that X causes Y.",
            cited_evidence_indices=[],
        )
        failures = detect_failures(task, output)
        assert FailureLabel.unsupported_biological_claim in failures

    def test_no_unsupported_claim_with_citations(self):
        task = make_task()
        output = make_output(
            answer_text="This confirms that X causes Y.",
            cited_evidence_indices=[0],
        )
        failures = detect_failures(task, output)
        assert FailureLabel.unsupported_biological_claim not in failures


class TestRunner:
    def test_dummy_run_completes(self):
        report = run_evaluation(mode="dummy", save_results=False)
        assert report.total_tasks >= 10
        assert 0.0 <= report.mean_overall <= 1.0

    def test_dummy_report_has_scores(self):
        report = run_evaluation(mode="dummy", save_results=False)
        assert len(report.scores) == report.total_tasks

    def test_dummy_failure_labels_populated(self):
        report = run_evaluation(mode="dummy", save_results=False)
        # Dummy baseline is intentionally naive — expect some failures
        assert len(report.failure_label_counts) > 0

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            run_evaluation(mode="garbage", save_results=False)  # type: ignore

    def test_manual_mode_missing_path_raises(self):
        with pytest.raises(ValueError, match="manual mode requires"):
            run_evaluation(mode="manual", save_results=False)

    def test_save_results_creates_file(self, tmp_path, monkeypatch):
        import bio_workflow_eval.runner as runner_module
        monkeypatch.setattr(runner_module, "OUTPUTS_DIR", tmp_path)
        report = run_evaluation(mode="dummy", save_results=True)
        saved_files = list(tmp_path.glob("*.json"))
        assert len(saved_files) == 1
        saved = json.loads(saved_files[0].read_text())
        assert "report" in saved
        assert "results" in saved
