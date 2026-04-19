"""
Optional API client for OpenRouter (supports OpenAI + Anthropic-compatible models).

Usage is entirely optional. If OPENROUTER_API_KEY is not set, this module
raises a clear error rather than failing silently.

Model outputs are parsed into ModelOutput objects.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from .schemas import BioTask, ModelOutput, NextAction

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-3-haiku"  # cheap, fast, good for evals

SYSTEM_PROMPT = """\
You are an expert biology researcher evaluating evidence for a scientific question.
For each task you will be given:
1. A question
2. A scenario
3. A set of evidence snippets

Your job: decide the best next action from these options:
- "answer": you have sufficient, non-conflicting evidence to answer
- "retrieve_more": more evidence is needed before a good answer is possible
- "defer": the question cannot be answered reliably with available information
- "use_tool": a specific tool or experiment is needed (e.g., run a database query)

Respond ONLY with valid JSON in this exact format:
{
  "predicted_action": "<answer|retrieve_more|defer|use_tool>",
  "answer_text": "<your answer if action is 'answer', else null>",
  "reasoning_trace": "<1-3 sentence explanation of your reasoning>",
  "cited_evidence_indices": [<list of 0-based indices of evidence snippets you relied on>]
}
"""


def _build_user_message(task: BioTask) -> str:
    evidence_block = "\n".join(
        f"[{i}] {snippet}" for i, snippet in enumerate(task.evidence)
    )
    return f"""Question: {task.question}

Scenario: {task.scenario}

Evidence:
{evidence_block}
"""


def call_openrouter(task: BioTask, model: str = DEFAULT_MODEL) -> ModelOutput:
    """
    Call OpenRouter API and parse the response into a ModelOutput.
    Raises EnvironmentError if OPENROUTER_API_KEY is not set.
    Raises httpx.HTTPError on API failure.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. "
            "Export it or use --mode dummy to run without an API key."
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(task)},
    ]

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/nihal-j/bio-workflow-eval",
                "X-Title": "bio-workflow-eval",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.1,  # low temp: we want consistent structured output
                "max_tokens": 512,
            },
        )
        response.raise_for_status()

    raw_content = response.json()["choices"][0]["message"]["content"]
    return _parse_response(task.task_id, raw_content)


def _parse_response(task_id: str, content: str) -> ModelOutput:
    """Parse the LLM JSON response into a ModelOutput, with graceful fallback."""
    try:
        # Strip markdown code fences if model wraps JSON in them
        cleaned = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data: dict[str, Any] = json.loads(cleaned)

        # Normalize action string
        action_str = data.get("predicted_action", "defer").lower().replace(" ", "_")
        try:
            action = NextAction(action_str)
        except ValueError:
            action = NextAction.defer

        return ModelOutput(
            task_id=task_id,
            predicted_action=action,
            answer_text=data.get("answer_text"),
            reasoning_trace=data.get("reasoning_trace"),
            cited_evidence_indices=data.get("cited_evidence_indices", []),
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        # Graceful fallback: defer if we can't parse
        return ModelOutput(
            task_id=task_id,
            predicted_action=NextAction.defer,
            answer_text=None,
            reasoning_trace=f"[parse error] raw response: {content[:200]}",
            cited_evidence_indices=[],
        )
