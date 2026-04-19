"""Tests for scoring metrics and individual dimension scorers."""

import pytest

from bio_workflow_eval.schemas import (
    BioTask,
    EvidenceStatus,
    FailureLabel,
    ModelOutput,
    NextAction,
)
from bio_workflow_eval.metrics import (
    compute_overall,
    evaluate,
    score_calibration,
    score_conflict_handling,
    score_decision_quality,
    score_evidence_grounding,
    score_unsupported_claim_penalty,
    SCORE_WEIGHTS,
)


def make_task(**overrides) -> BioTask:
    base = dict(
        task_id="t001",
        domain="test",
        question="Is X true?",
        scenario="A lab asks whether X is true.",
        evidence=["Snippet A.", "Snippet B.", "Snippet C."],
        evidence_status=EvidenceStatus.sufficient,
        correct_action=NextAction.answer,
        gold_answer="Yes, X is true.",
        gold_reasoning="Snippet A and B support X.",
        expected_failure_labels=[],
    )
    base.update(overrides)
    return BioTask(**base)


def make_output(**overrides) -> ModelOutput:
    base = dict(
        task_id="t001",
        predicted_action=NextAction.answer,
        answer_text="Yes, X is true.",
        reasoning_trace="Based on Snippet A.",
        cited_evidence_indices=[0, 1],
    )
    base.update(overrides)
    return ModelOutput(**base)


class TestDecisionQuality:
    def test_correct_action_full_score(self):
        task = make_task(correct_action=NextAction.answer)
        output = make_output(predicted_action=NextAction.answer)
        assert score_decision_quality(task, output) == 1.0

    def test_wrong_action_zero_score(self):
        task = make_task(correct_action=NextAction.answer)
        output = make_output(predicted_action=NextAction.use_tool)
        assert score_decision_quality(task, output) == 0.0

    def test_partial_credit_insufficient_defer(self):
        task = make_task(
            correct_action=NextAction.answer,
            evidence_status=EvidenceStatus.insufficient,
        )
        output = make_output(predicted_action=NextAction.defer)
        score = score_decision_quality(task, output)
        assert 0.0 < score < 1.0

    def test_partial_credit_conflicting_retrieve(self):
        task = make_task(
            correct_action=NextAction.answer,
            evidence_status=EvidenceStatus.conflicting,
        )
        output = make_output(predicted_action=NextAction.retrieve_more)
        score = score_decision_quality(task, output)
        assert 0.0 < score < 1.0


class TestEvidenceGrounding:
    def test_no_citations_zero_score(self):
        task = make_task()
        output = make_output(cited_evidence_indices=[])
        assert score_evidence_grounding(task, output) == 0.0

    def test_valid_citations_positive(self):
        task = make_task()
        output = make_output(cited_evidence_indices=[0])
        assert score_evidence_grounding(task, output) > 0.0

    def test_all_citations_max_score(self):
        task = make_task()
        output = make_output(cited_evidence_indices=[0, 1, 2])
        assert score_evidence_grounding(task, output) == 1.0

    def test_out_of_bounds_indices_ignored(self):
        task = make_task()
        output = make_output(cited_evidence_indices=[99, 100])
        assert score_evidence_grounding(task, output) == 0.0


class TestConflictHandling:
    def test_non_conflicting_full_credit(self):
        task = make_task(evidence_status=EvidenceStatus.sufficient)
        output = make_output(predicted_action=NextAction.answer)
        assert score_conflict_handling(task, output) == 1.0

    def test_conflicting_defer_full_credit(self):
        task = make_task(evidence_status=EvidenceStatus.conflicting)
        output = make_output(predicted_action=NextAction.defer)
        assert score_conflict_handling(task, output) == 1.0

    def test_conflicting_answer_zero_score(self):
        task = make_task(evidence_status=EvidenceStatus.conflicting)
        output = make_output(predicted_action=NextAction.answer)
        assert score_conflict_handling(task, output) == 0.0


class TestOverallScore:
    def test_weights_sum_to_one(self):
        total = sum(SCORE_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6

    def test_perfect_score(self):
        scores = {k: 1.0 for k in SCORE_WEIGHTS}
        assert compute_overall(scores) == pytest.approx(1.0)

    def test_zero_score(self):
        scores = {k: 0.0 for k in SCORE_WEIGHTS}
        assert compute_overall(scores) == pytest.approx(0.0)


class TestEvaluate:
    def test_returns_eval_score(self):
        task = make_task()
        output = make_output()
        score = evaluate(task, output)
        assert 0.0 <= score.overall_score <= 1.0

    def test_correct_action_scores_well(self):
        task = make_task(correct_action=NextAction.answer)
        output = make_output(predicted_action=NextAction.answer, cited_evidence_indices=[0, 1, 2])
        score = evaluate(task, output)
        assert score.overall_score >= 0.7

    def test_wrong_action_no_citations_scores_poorly(self):
        task = make_task(correct_action=NextAction.answer)
        output = make_output(
            predicted_action=NextAction.defer,
            cited_evidence_indices=[],
        )
        score = evaluate(task, output)
        assert score.overall_score < 0.6
