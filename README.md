# bio-workflow-eval

A small evaluation harness for biology/scientific workflow reasoning tasks.

This project tests whether a model or workflow can make good **process decisions** in multi-step biology reasoning — not just produce correct final answers.

---

## What this evaluates

Most biology QA benchmarks ask "what is the answer?" This project asks:

- Should the system **answer**, **retrieve more evidence**, **defer**, or **use a tool**?
- Does it ground its answer in the right evidence, or just say something plausible?
- Does it detect when evidence is conflicting, or charge ahead anyway?
- Is it correctly calibrated about uncertainty?

These are the kinds of failures that matter in scientific agents, retrieval-augmented workflows, and AI-for-science systems.

---

## Quick start

### 1. Install

```bash
# Using uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Or standard pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Run the CLI

```bash
# Evaluate the dummy baseline (no API key needed)
bio-eval run --mode dummy

# Show the benchmark task list
bio-eval tasks

# Show a saved result
bio-eval show-report outputs/run_YYYYMMDD_HHMMSS_xxxxxx.json
```

### 3. Run with a real model (OpenRouter)

```bash
export OPENROUTER_API_KEY=sk-or-...
bio-eval run --mode api --model anthropic/claude-3-haiku
```

### 4. Launch the Streamlit viewer

```bash
streamlit run app/streamlit_app.py
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Project structure

```
bio-workflow-eval/
  tasks/benchmark.json       # 16 hand-crafted biology reasoning tasks
  src/bio_workflow_eval/
    schemas.py               # Pydantic models: BioTask, ModelOutput, EvalScore
    loader.py                # Load and validate benchmark tasks
    metrics.py               # Rubric-based scoring (5 dimensions)
    failures.py              # Failure label detection
    baselines.py             # DummyBaseline, ManualBaseline
    runner.py                # Core evaluation pipeline
    cli.py                   # Typer CLI (run, tasks, show-report)
    api_clients.py           # Optional OpenRouter API integration
  app/streamlit_app.py       # 3-page Streamlit viewer
  tests/                     # Pytest tests (loader, metrics, runner)
  outputs/                   # Saved evaluation JSONs
```

---

## Evaluation dimensions

| Dimension | What it measures |
|-----------|-----------------|
| `decision_quality` | Did the model choose the right action (answer/retrieve/defer/use_tool)? |
| `evidence_grounding` | Did it cite the right evidence snippets? |
| `conflict_handling` | Did it handle conflicting evidence appropriately? |
| `unsupported_claim_penalty` | Did it make confident biological claims without evidence? |
| `calibration_or_defer_quality` | Was it appropriately cautious under uncertainty? |
| `overall_score` | Weighted average (weights in `metrics.py`) |

Scoring is rubric-based and transparent — the logic is readable in `metrics.py`.

---

## Failure labels

The harness also labels *why* a model failed, not just that it did:

| Label | Meaning |
|-------|---------|
| `answered_too_early` | Gave an answer when evidence was insufficient/conflicting |
| `ignored_conflicting_evidence` | Answered confidently despite conflicting snippets |
| `wrong_next_action` | Chose a clearly wrong workflow action |
| `unsupported_biological_claim` | Made a confident claim with no evidence citation |
| `shallow_evidence_use` | Only engaged with a fraction of available evidence |

---

## Benchmark domains

The 16 tasks span:
- **Cancer biology** (KRAS, BRCA2, DLBCL, SCLC TMB)
- **Single-cell biology** (T cell annotation, HSC trajectories, ambient RNA)
- **Pathway/mechanism** (PTEN phosphorylation, WNT signaling, mTOR)
- **Gene expression** (MYC perturbation, RNA-seq stats)
- **Experimental design** (CRISPR validation, scRNA-seq vs. bulk, EMT scoring)
- **Pathology** (lymphoma IHC interpretation)

---

## Limitations

This is a prototype, not a validated benchmark. Be honest about what it is:

- Tasks are hand-crafted synthetic/adapted snippets — not curated from live literature
- Scoring is heuristic and rule-based — not peer-reviewed
- The dummy baseline is intentionally bad and is not a scientific comparison point
- 16 tasks is too small for reliable statistical claims
- Failure labels use keyword heuristics and can miss edge cases

This is a methods prototype appropriate for internal research, presentation, and iteration — not for publication as a primary evaluation.

---

## Extending this

To add tasks: edit `tasks/benchmark.json` following the schema in `schemas.py`.

To add a new scoring dimension: add a function to `metrics.py` and update `SCORE_WEIGHTS`.

To add a new baseline: implement `predict(task: BioTask) -> ModelOutput` and wire it into `runner.py`.

To support a new model API: add a client to `api_clients.py`.

---

## Why this kind of eval matters

Most LLM benchmarks reward producing the right answer. Scientific and clinical workflows also need models that know *when not to answer*, how to handle incomplete or conflicting evidence, and when to request more information or invoke a tool. This harness makes those behaviors explicitly testable — even at small prototype scale.

---

*Author: nihal-j*
