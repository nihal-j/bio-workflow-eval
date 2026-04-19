"""
Core runner: loads tasks, runs a baseline or model, scores each task, and saves results.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from .baselines import DummyBaseline, ManualBaseline
from .loader import load_tasks
from .metrics import evaluate
from .schemas import BioTask, EvalReport, EvalScore, ModelOutput, RunResult

Mode = Literal["dummy", "manual", "api"]

OUTPUTS_DIR = Path(__file__).parents[2] / "outputs"


def run_evaluation(
    mode: Mode = "dummy",
    tasks_path: Path | str | None = None,
    manual_outputs_path: Path | str | None = None,
    api_model: str = "anthropic/claude-3-haiku",
    save_results: bool = True,
) -> EvalReport:
    """
    Main evaluation entry point.

    Returns an EvalReport with per-task scores and aggregate stats.
    """
    tasks = load_tasks(tasks_path)
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    outputs: list[ModelOutput] = _get_outputs(mode, tasks, manual_outputs_path, api_model)
    results: list[RunResult] = _score_all(tasks, outputs)

    report = _build_report(run_id, mode, api_model, results)

    if save_results:
        _save_report(report, results)

    return report


def _get_outputs(
    mode: Mode,
    tasks: list[BioTask],
    manual_outputs_path: Path | str | None,
    api_model: str,
) -> list[ModelOutput]:
    """Dispatch to the right prediction source based on mode."""
    if mode == "dummy":
        baseline = DummyBaseline()
        return [baseline.predict(task) for task in tasks]

    if mode == "manual":
        if not manual_outputs_path:
            raise ValueError("manual mode requires --manual-outputs <path>")
        baseline = ManualBaseline(manual_outputs_path)
        return [baseline.predict(task) for task in tasks]

    if mode == "api":
        from .api_clients import call_openrouter
        return [call_openrouter(task, model=api_model) for task in tasks]

    raise ValueError(f"Unknown mode: {mode!r}. Use dummy, manual, or api.")


def _score_all(tasks: list[BioTask], outputs: list[ModelOutput]) -> list[RunResult]:
    """Match outputs to tasks by task_id and score each pair."""
    output_map = {o.task_id: o for o in outputs}
    results = []
    for task in tasks:
        output = output_map.get(task.task_id)
        if output is None:
            continue
        score = evaluate(task, output)
        results.append(RunResult(task=task, output=output, score=score))
    return results


def _build_report(
    run_id: str,
    mode: str,
    api_model: str,
    results: list[RunResult],
) -> EvalReport:
    """Aggregate per-task scores into a report."""
    from collections import Counter

    scores = [r.score for r in results]
    n = len(scores)

    if n == 0:
        raise ValueError("No results to aggregate — check that task IDs match outputs.")

    model_name = api_model if mode == "api" else mode

    failure_counts: Counter[str] = Counter()
    for score in scores:
        for label in score.failure_labels:
            failure_counts[label.value] += 1

    return EvalReport(
        run_id=run_id,
        mode=mode,
        model_name=model_name,
        scores=scores,
        mean_overall=round(sum(s.overall_score for s in scores) / n, 4),
        mean_decision_quality=round(sum(s.decision_quality for s in scores) / n, 4),
        mean_evidence_grounding=round(sum(s.evidence_grounding for s in scores) / n, 4),
        mean_conflict_handling=round(sum(s.conflict_handling for s in scores) / n, 4),
        failure_label_counts=dict(failure_counts),
        total_tasks=n,
    )


def _save_report(report: EvalReport, results: list[RunResult]) -> Path:
    """Write report JSON + per-task results to outputs/."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    out_path = OUTPUTS_DIR / f"{report.run_id}.json"

    payload = {
        "report": report.model_dump(),
        "results": [
            {
                "task_id": r.task.task_id,
                "domain": r.task.domain,
                "correct_action": r.task.correct_action.value,
                "predicted_action": r.output.predicted_action.value,
                "answer_text": r.output.answer_text,
                "reasoning_trace": r.output.reasoning_trace,
                "cited_evidence_indices": r.output.cited_evidence_indices,
                "score": r.score.model_dump(),
            }
            for r in results
        ],
    }

    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
