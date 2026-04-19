"""
Baseline implementations for evaluating without a real model.

- DummyBaseline: keyword/heuristic-based, answer-first, naive
- ManualBaseline: loads pre-written outputs from a JSON file for interactive review
"""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import BioTask, EvidenceStatus, ModelOutput, NextAction

CONFLICT_KEYWORDS = ["however", "in contrast", "but", "contradicts", "although", "conversely"]
INSUFFICIENT_KEYWORDS = ["limited", "unclear", "no data", "unknown", "not enough", "insufficient"]


class DummyBaseline:
    """
    A deliberately naive baseline. It mimics the kind of model that:
    - answers whenever possible (overconfident)
    - uses simple keyword matching for conflict/insufficient detection
    - never retrieves or uses tools

    This is intentionally bad — it gives us something to fail against.
    """

    name = "dummy_baseline"

    def predict(self, task: BioTask) -> ModelOutput:
        combined_evidence = " ".join(task.evidence).lower()

        predicted_action = self._decide_action(task, combined_evidence)
        cited = self._choose_evidence(task, combined_evidence)
        answer_text = self._generate_answer(task) if predicted_action == NextAction.answer else None

        return ModelOutput(
            task_id=task.task_id,
            predicted_action=predicted_action,
            answer_text=answer_text,
            reasoning_trace="[dummy] keyword-based heuristic",
            cited_evidence_indices=cited,
        )

    def _decide_action(self, task: BioTask, evidence_text: str) -> NextAction:
        has_conflict = any(kw in evidence_text for kw in CONFLICT_KEYWORDS)
        has_insufficient = any(kw in evidence_text for kw in INSUFFICIENT_KEYWORDS)

        if has_conflict:
            # Dummy still tries to answer despite conflict — this is a known failure
            return NextAction.answer
        if has_insufficient:
            return NextAction.defer
        return NextAction.answer

    def _choose_evidence(self, task: BioTask, evidence_text: str) -> list[int]:
        question_words = set(task.question.lower().split())
        scored = []
        for i, snippet in enumerate(task.evidence):
            overlap = len(question_words & set(snippet.lower().split()))
            scored.append((i, overlap))
        scored.sort(key=lambda x: -x[1])
        # Cite only the top-1 match — intentionally shallow
        return [scored[0][0]] if scored else []

    def _generate_answer(self, task: BioTask) -> str:
        return f"Based on the evidence, {task.gold_answer or 'the answer is not immediately clear'}."


class ManualBaseline:
    """
    Load pre-written model outputs from a JSON file.
    Useful for evaluating pasted LLM outputs or hand-crafted responses.

    The file should be a JSON array of ModelOutput-compatible dicts.
    """

    name = "manual"

    def __init__(self, outputs_path: Path | str):
        self.outputs_path = Path(outputs_path)
        self._outputs: dict[str, ModelOutput] = {}
        self._load()

    def _load(self) -> None:
        if not self.outputs_path.exists():
            raise FileNotFoundError(f"Manual outputs file not found: {self.outputs_path}")
        raw = json.loads(self.outputs_path.read_text())
        for item in raw:
            output = ModelOutput.model_validate(item)
            self._outputs[output.task_id] = output

    def predict(self, task: BioTask) -> ModelOutput:
        if task.task_id not in self._outputs:
            raise KeyError(f"No manual output for task '{task.task_id}'")
        return self._outputs[task.task_id]
