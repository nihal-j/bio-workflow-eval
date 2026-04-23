# bio-workflow-eval

A small Python tool to test whether AI models can make good scientific decisions. 

Most biology benchmarks ask "what is the answer?". But real scientific workflows don't work like that. If a model looks at evidence, it should know when to say "I need more data" or "these papers are conflicting" rather than just guessing. This project tests exactly that.

The tool tests if a model can make four decisions:
1. `answer`: It has enough solid evidence to answer.
2. `retrieve_more`: It needs more data before answering.
3. `defer`: The evidence is too messy or conflicting to answer.
4. `use_tool`: It needs a specific experiment or tool.

---

## Structure

```
bio-workflow-eval/
├── tasks/
│   └── benchmark.json        # 16 hand-written biology scenarios
├── src/bio_workflow_eval/    # The main Python code
│   ├── baselines.py          # A dumb fake model used for testing
│   ├── metrics.py            # The scoring logic
│   ├── runner.py             # Runs the test and saves the results
│   ├── cli.py                # Terminal commands
│   ├── schemas.py            # Code shapes
│   ├── failures.py           # Assigns badges like "answered_too_early"
│   └── api_clients.py        # Hooks up to real AIs via OpenRouter 
├── app/
│   └── streamlit_app.py      # The web dashboard you see at localhost:8501
└── tests/                    # Tests to prove the code works
```

---

## Setup

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

---

## Running it

```bash
# see the 16 biology tests
bio-eval tasks

# run the test using the fake dumb model (no API key needed)
# (This is what you probably ran. It's meant to fail to prove the grader works.)
bio-eval run --mode dummy

# run the test on a real AI (like Claude or GPT)
export OPENROUTER_API_KEY=sk-or-...
bio-eval run --mode api --model anthropic/claude-3-haiku

# launch the web dashboard to see the scores
streamlit run app/streamlit_app.py
```

---

## How scoring works

When an AI answers a scenario, it gets graded on 5 things:
- **Decision**: Did it pick the right action (e.g. `defer`)?
- **Evidence**: Did it cite the specific evidence snippets?
- **Conflict**: Did it handle conflicting evidence properly?
- **Claims**: Did it make bold claims without proof?
- **Calibration**: Did it express the right amount of uncertainty?

If it messes up, it gets tagged with labels like `answered_too_early` or `ignored_conflicting_evidence`.

---

## The scenarios

There are 16 fake biology scenarios covering cancer biology, single-cell analysis, gene expression, and pathology. 

Some scenarios have enough evidence to answer. Some don't. Some have evidence that contradicts itself. The goal is to see if the AI can tell the difference.

---

## Limitations

- 16 tasks is a tiny sample size.
- The scoring logic is rough and not scientifically validated.
- The dummy mode is intentionally bad just to test the software.
