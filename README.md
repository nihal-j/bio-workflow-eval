# bio-workflow-eval

A small eval harness for testing whether a model can reason well across multi-step biology tasks — not just produce a correct answer, but decide *what to do next* given incomplete or conflicting evidence.

---

## Why

Most biology benchmarks ask "what is the answer?" Real scientific workflows don't work like that. A system looking at evidence should sometimes say *I need more data* or *this is conflicting, I can't commit* instead of always trying to answer. This project makes those decisions testable.

The four decisions every task asks for:

| Action | When |
|--------|------|
| `answer` | Evidence is sufficient and consistent |
| `retrieve_more` | More data is needed before answering |
| `defer` | Evidence is conflicting or too uncertain |
| `use_tool` | A specific tool or experiment is required |

---

## Structure

```
bio-workflow-eval/
├── tasks/
│   └── benchmark.json        # 16 hand-written biology tasks
├── src/bio_workflow_eval/
│   ├── schemas.py            # data types (task, output, score)
│   ├── loader.py             # loads + validates benchmark.json
│   ├── metrics.py            # scoring (5 dimensions, explicit weights)
│   ├── failures.py           # labels why a model failed, not just that it did
│   ├── baselines.py          # dummy baseline + manual output loader
│   ├── runner.py             # load → predict → score → save
│   ├── cli.py                # terminal interface
│   └── api_clients.py        # optional: OpenRouter API support
├── app/
│   └── streamlit_app.py      # web viewer (localhost:8501)
└── tests/                    # 39 pytest tests
```

---

## Setup

```bash
# requires Python 3.11+, uv recommended
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

---

## Running it

```bash
# see the task list
bio-eval tasks

# run the dummy baseline (no API key needed)
bio-eval run --mode dummy

# run with a real model via OpenRouter
export OPENROUTER_API_KEY=sk-or-...
bio-eval run --mode api --model anthropic/claude-3-haiku

# load a previous result
bio-eval show-report outputs/run_20240419_123456_abc123.json

# web UI
streamlit run app/streamlit_app.py

# tests
pytest tests/ -v
```

---

## How scoring works

Each model output is scored across five dimensions (all 0–1, weights in `metrics.py`):

- **decision_quality** — did it pick the right action?
- **evidence_grounding** — did it cite relevant snippets?
- **conflict_handling** — did it handle conflicting evidence correctly?
- **unsupported_claim_penalty** — did it make confident claims without evidence?
- **calibration_or_defer_quality** — was it appropriately uncertain?

On top of that, `failures.py` labels *why* it failed: `answered_too_early`, `ignored_conflicting_evidence`, `shallow_evidence_use`, etc. The failure labels are meant to be more useful than a single score.

---

## The benchmark

16 tasks across 6 domains, all hand-written:

- cancer biology (KRAS, BRCA2, DLBCL, SCLC TMB)
- single-cell biology (T cell annotation, HSC trajectories, ambient RNA)
- pathway / mechanism (PTEN, WNT, mTOR)
- gene expression (MYC perturbation, RNA-seq replication)
- experimental design (CRISPR validation, scRNA-seq vs. bulk)
- pathology (lymphoma IHC interpretation)

Each task has 2–5 evidence snippets, a correctness label (`sufficient` / `insufficient` / `conflicting`), a gold action, and expected failure modes for a model that reasons poorly.

---

## Limitations

This is a prototype. Be honest about what it is:

- 16 tasks is too small for statistical claims
- scoring is heuristic, not peer-reviewed
- the dummy baseline is intentionally bad — it is not a scientific comparison
- failure labels use keyword heuristics and will miss edge cases

Good for: internal research, exploring failure modes, extending toward larger evals or retrieval-augmented setups.
