"""
Failure analysis: labels common reasoning errors.

The goal is to explain *why* a model failed, not just that it did.
These labels help identify systematic weaknesses across a run.
"""

from __future__ import annotations

from .schemas import BioTask, EvidenceStatus, FailureLabel, ModelOutput, NextAction

# Simple keyword patterns that hint at a biological claim being made
BIO_CLAIM_KEYWORDS = [
    "therefore", "this means", "the result is", "confirms that",
    "demonstrates that", "shows that", "causes", "leads to",
    "is responsible for", "activates", "inhibits", "upregulates", "downregulates",
]

CONFLICT_INDICATORS = ["however", "in contrast", "but", "although", "conversely", "contradicts"]


def detect_failures(task: BioTask, output: ModelOutput) -> list[FailureLabel]:
    """
    Heuristically label failures in model output relative to the task.
    Returns a (possibly empty) list of FailureLabel values.
    """
    labels: list[FailureLabel] = []

    labels.extend(_check_answered_too_early(task, output))
    labels.extend(_check_ignored_conflict(task, output))
    labels.extend(_check_wrong_next_action(task, output))
    labels.extend(_check_unsupported_claim(task, output))
    labels.extend(_check_shallow_evidence(task, output))

    return labels


def _check_answered_too_early(task: BioTask, output: ModelOutput) -> list[FailureLabel]:
    """Model answered when it should have retrieved more or deferred."""
    if output.predicted_action == NextAction.answer:
        if task.evidence_status in (EvidenceStatus.insufficient, EvidenceStatus.conflicting):
            return [FailureLabel.answered_too_early]
    return []


def _check_ignored_conflict(task: BioTask, output: ModelOutput) -> list[FailureLabel]:
    """Model gave a confident answer despite conflicting evidence."""
    if task.evidence_status == EvidenceStatus.conflicting:
        if output.predicted_action == NextAction.answer:
            return [FailureLabel.ignored_conflicting_evidence]
    return []


def _check_wrong_next_action(task: BioTask, output: ModelOutput) -> list[FailureLabel]:
    """Model chose an action that isn't close to the gold action."""
    if output.predicted_action == task.correct_action:
        return []

    # Some near-misses are okay — only flag clear wrong turns
    bad_pairs = {
        (NextAction.answer, NextAction.defer),
        (NextAction.answer, NextAction.retrieve_more),
        (NextAction.defer, NextAction.use_tool),
        (NextAction.retrieve_more, NextAction.answer),
    }
    if (output.predicted_action, task.correct_action) in bad_pairs:
        return [FailureLabel.wrong_next_action]

    return []


def _check_unsupported_claim(task: BioTask, output: ModelOutput) -> list[FailureLabel]:
    """
    Model made a confident biological claim without grounding in evidence.
    Heuristic: uses claim-language keywords + has no cited evidence.
    """
    if output.predicted_action != NextAction.answer:
        return []

    answer = (output.answer_text or "").lower()
    has_claim_language = any(kw in answer for kw in BIO_CLAIM_KEYWORDS)
    n_valid_citations = sum(
        1 for i in output.cited_evidence_indices if 0 <= i < len(task.evidence)
    )

    if has_claim_language and n_valid_citations == 0:
        return [FailureLabel.unsupported_biological_claim]

    return []


def _check_shallow_evidence(task: BioTask, output: ModelOutput) -> list[FailureLabel]:
    """
    Model cited very few snippets relative to what was available.
    Signals surface-level engagement with the evidence.
    """
    n_evidence = len(task.evidence)
    if n_evidence <= 1:
        return []  # Nothing to be shallow about

    n_valid_citations = sum(
        1 for i in output.cited_evidence_indices if 0 <= i < n_evidence
    )

    # Cited less than half of available evidence when answering
    if output.predicted_action == NextAction.answer and n_valid_citations < n_evidence / 2:
        return [FailureLabel.shallow_evidence_use]

    return []
