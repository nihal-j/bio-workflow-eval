"""
Data schemas for tasks, model outputs, and evaluation results.
Using Pydantic for runtime validation and clean serialization.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EvidenceStatus(str, Enum):
    sufficient = "sufficient"
    insufficient = "insufficient"
    conflicting = "conflicting"


class NextAction(str, Enum):
    answer = "answer"
    retrieve_more = "retrieve_more"
    defer = "defer"
    use_tool = "use_tool"


class FailureLabel(str, Enum):
    answered_too_early = "answered_too_early"
    ignored_conflicting_evidence = "ignored_conflicting_evidence"
    wrong_next_action = "wrong_next_action"
    unsupported_biological_claim = "unsupported_biological_claim"
    shallow_evidence_use = "shallow_evidence_use"


class BioTask(BaseModel):
    """A single evaluation task in the benchmark."""

    task_id: str
    domain: str
    question: str
    scenario: str
    evidence: list[str] = Field(min_length=1, max_length=5)
    evidence_status: EvidenceStatus
    correct_action: NextAction
    gold_answer: Optional[str] = None
    gold_reasoning: str
    expected_failure_labels: list[FailureLabel] = Field(default_factory=list)


class ModelOutput(BaseModel):
    """What the model (or baseline) said for a given task."""

    task_id: str
    predicted_action: NextAction
    answer_text: Optional[str] = None
    reasoning_trace: Optional[str] = None
    cited_evidence_indices: list[int] = Field(default_factory=list)


class EvalScore(BaseModel):
    """Rubric-based score for a single task evaluation."""

    task_id: str
    decision_quality: float = Field(ge=0.0, le=1.0)
    evidence_grounding: float = Field(ge=0.0, le=1.0)
    conflict_handling: float = Field(ge=0.0, le=1.0)
    unsupported_claim_penalty: float = Field(ge=0.0, le=1.0)
    calibration_or_defer_quality: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    failure_labels: list[FailureLabel] = Field(default_factory=list)
    notes: str = ""


class RunResult(BaseModel):
    """Full result of evaluating one model output against one task."""

    task: BioTask
    output: ModelOutput
    score: EvalScore


class EvalReport(BaseModel):
    """Aggregated report for a full evaluation run."""

    run_id: str
    mode: str
    model_name: str
    scores: list[EvalScore]
    mean_overall: float
    mean_decision_quality: float
    mean_evidence_grounding: float
    mean_conflict_handling: float
    failure_label_counts: dict[str, int]
    total_tasks: int
