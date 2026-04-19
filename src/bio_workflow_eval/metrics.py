"""
Scoring logic for evaluating model outputs against benchmark tasks.

Philosophy: simple, transparent, rubric-based. Not perfect — and that's fine.
Each dimension is scored 0.0 to 1.0. The overall score is a weighted average.
"""

from __future__ import annotations

from .failures import detect_failures
from .schemas import BioTask, EvalScore, ModelOutput, NextAction, EvidenceStatus

# Weights for overall score. These are explicit and easy to adjust.
SCORE_WEIGHTS = {
    "decision_quality": 0.35,
    "evidence_grounding": 0.25,
    "conflict_handling": 0.20,
    "unsupported_claim_penalty": 0.10,
    "calibration_or_defer_quality": 0.10,
}


def score_decision_quality(task: BioTask, output: ModelOutput) -> float:
    """
    Did the model choose the right action?
    Full credit for correct action, partial for related actions.
    """
    if output.predicted_action == task.correct_action:
        return 1.0

    # Partial credit: defer is acceptable when insufficient evidence
    if task.evidence_status == EvidenceStatus.insufficient:
        if output.predicted_action in (NextAction.defer, NextAction.retrieve_more):
            return 0.5

    # Partial credit: retrieve_more acceptable when conflicting
    if task.evidence_status == EvidenceStatus.conflicting:
        if output.predicted_action == NextAction.retrieve_more:
            return 0.4

    return 0.0


def score_evidence_grounding(task: BioTask, output: ModelOutput) -> float:
    """
    Did the model cite relevant evidence snippets?
    Rewards citing ≥ 1 relevant snippet. Penalizes citing nothing.
    """
    if not task.evidence:
        return 1.0  # edge case: no evidence available

    cited = output.cited_evidence_indices
    n_evidence = len(task.evidence)
    valid_citations = [i for i in cited if 0 <= i < n_evidence]

    if not valid_citations:
        return 0.0

    # Normalize: reward for citing more snippets up to all of them
    coverage = len(valid_citations) / n_evidence
    return min(1.0, coverage * 1.5)  # slight bonus for citing most evidence


def score_conflict_handling(task: BioTask, output: ModelOutput) -> float:
    """
    How well did the model handle conflicting evidence?
    Only meaningful when evidence_status == conflicting.
    """
    if task.evidence_status != EvidenceStatus.conflicting:
        return 1.0  # N/A — full credit, doesn't apply

    # If evidence conflicts and the model defers or retrieves more: good
    if output.predicted_action in (NextAction.defer, NextAction.retrieve_more):
        return 1.0

    # If model answers confidently with conflicting evidence present: bad
    if output.predicted_action == NextAction.answer:
        return 0.0

    return 0.5


def score_unsupported_claim_penalty(task: BioTask, output: ModelOutput) -> float:
    """
    Did the model make a confident claim without grounding evidence?
    This is the "plausible but wrong" failure mode.

    Score: 1.0 = no unsupported claims detected, 0.0 = likely unsupported.
    """
    if output.predicted_action != NextAction.answer:
        return 1.0  # Not answering — no claim penalty applies

    if not output.answer_text:
        return 1.0  # No answer text — nothing to penalize

    cited = output.cited_evidence_indices
    n_evidence = len(task.evidence)
    valid_citations = [i for i in cited if 0 <= i < n_evidence]

    # If answered with zero citation → likely unsupported
    if not valid_citations:
        return 0.0

    return 1.0


def score_calibration(task: BioTask, output: ModelOutput) -> float:
    """
    Was the model appropriately cautious when it should be?
    Rewards deferring or asking for more info when evidence is insufficient/conflicting.
    Penalizes overconfident answers in uncertain conditions.
    """
    is_uncertain = task.evidence_status in (EvidenceStatus.insufficient, EvidenceStatus.conflicting)

    if is_uncertain and output.predicted_action == NextAction.answer:
        return 0.0  # Overconfident answer in uncertain scenario

    if not is_uncertain and output.predicted_action == NextAction.defer:
        return 0.5  # Unnecessarily cautious, but not the worst

    if is_uncertain and output.predicted_action in (NextAction.defer, NextAction.retrieve_more):
        return 1.0  # Correctly cautious

    return 1.0  # Sufficient evidence, answer given — fine


def compute_overall(scores: dict[str, float]) -> float:
    """Weighted average of dimension scores."""
    total = sum(SCORE_WEIGHTS[k] * scores[k] for k in SCORE_WEIGHTS)
    return round(total, 4)


def evaluate(task: BioTask, output: ModelOutput) -> EvalScore:
    """
    Run all scoring dimensions and return an EvalScore.
    This is the main entry point for scoring.
    """
    dim_scores = {
        "decision_quality": score_decision_quality(task, output),
        "evidence_grounding": score_evidence_grounding(task, output),
        "conflict_handling": score_conflict_handling(task, output),
        "unsupported_claim_penalty": score_unsupported_claim_penalty(task, output),
        "calibration_or_defer_quality": score_calibration(task, output),
    }

    failure_labels = detect_failures(task, output)
    overall = compute_overall(dim_scores)

    return EvalScore(
        task_id=task.task_id,
        failure_labels=failure_labels,
        overall_score=overall,
        notes=f"action={output.predicted_action.value}, gold={task.correct_action.value}",
        **dim_scores,
    )
