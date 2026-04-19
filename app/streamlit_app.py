"""
Streamlit viewer for bio-workflow-eval results.

Run with: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from bio_workflow_eval.loader import load_tasks, task_summary
from bio_workflow_eval.runner import run_evaluation
from bio_workflow_eval.schemas import EvalReport

OUTPUTS_DIR = Path(__file__).parents[1] / "outputs"

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="bio-workflow-eval",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (minimal, clean) ────────────────────────────────────────────
st.markdown(
    """
    <style>
      .metric-card {
        background: #1e2530;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        border: 1px solid #2d3748;
      }
      .metric-value { font-size: 2rem; font-weight: 700; color: #63b3ed; }
      .metric-label { font-size: 0.8rem; color: #a0aec0; margin-top: 4px; }
      .failure-chip {
        display: inline-block;
        background: #742a2a;
        color: #fed7d7;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin: 2px;
      }
      .tag-sufficient { color: #68d391; }
      .tag-conflicting { color: #fc8181; }
      .tag-insufficient { color: #f6ad55; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧬 bio-workflow-eval")
    st.caption("Biology reasoning evaluation harness")
    st.divider()

    page = st.radio("View", ["📊 Run & Results", "📋 Task Browser", "🔍 Inspect Example"])
    st.divider()

    run_mode = st.selectbox("Baseline mode", ["dummy", "api"])
    if run_mode == "api":
        api_model = st.text_input("OpenRouter model", value="anthropic/claude-3-haiku")
    else:
        api_model = "dummy"

    if st.button("▶  Run Evaluation", use_container_width=True, type="primary"):
        with st.spinner("Running evaluation..."):
            try:
                report = run_evaluation(mode=run_mode, api_model=api_model, save_results=True)
                st.session_state["report"] = report
                st.session_state["run_mode"] = run_mode
                st.success(f"Done — {report.total_tasks} tasks scored")
            except EnvironmentError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # Load existing report
    saved = sorted(OUTPUTS_DIR.glob("*.json"), reverse=True)
    if saved:
        st.caption("Or load a saved run:")
        selected = st.selectbox("Saved runs", [p.name for p in saved], label_visibility="collapsed")
        if st.button("Load", use_container_width=True):
            raw = json.loads((OUTPUTS_DIR / selected).read_text())
            report_data = raw["report"]
            st.session_state["raw_results"] = raw["results"]
            st.session_state["loaded_report_data"] = report_data
            st.success("Loaded")


# ─── Helper: get active report data ─────────────────────────────────────────
def get_report_and_results():
    """Return (report_dict, results_list) from session state, however available."""
    if "report" in st.session_state:
        r: EvalReport = st.session_state["report"]
        # Reconstruct results from the saved file
        saved = sorted(OUTPUTS_DIR.glob(f"{r.run_id}*.json"), reverse=True)
        results = []
        if saved:
            raw = json.loads(saved[0].read_text())
            results = raw.get("results", [])
        return r.model_dump(), results
    if "loaded_report_data" in st.session_state:
        return st.session_state["loaded_report_data"], st.session_state.get("raw_results", [])
    return None, None


# ─── Page: Run & Results ─────────────────────────────────────────────────────
if page == "📊 Run & Results":
    st.header("Evaluation Results")
    report_data, results = get_report_and_results()

    if not report_data:
        st.info("Run an evaluation using the sidebar or load a saved run to see results.")
        st.stop()

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (c1, "Overall Score", f"{report_data['mean_overall']:.3f}"),
        (c2, "Decision Quality", f"{report_data['mean_decision_quality']:.3f}"),
        (c3, "Evidence Grounding", f"{report_data['mean_evidence_grounding']:.3f}"),
        (c4, "Conflict Handling", f"{report_data['mean_conflict_handling']:.3f}"),
    ]
    for col, label, val in metrics:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # Failure label bar chart
    failures = report_data.get("failure_label_counts", {})
    if failures:
        st.subheader("Failure Label Distribution")
        st.bar_chart(failures)

    st.divider()

    # Per-task table
    if results:
        st.subheader("Per-Task Scores")
        rows = []
        for r in results:
            s = r["score"]
            rows.append({
                "Task ID": r["task_id"],
                "Domain": r["domain"],
                "Gold Action": r["correct_action"],
                "Predicted": r["predicted_action"],
                "Decision": round(s["decision_quality"], 2),
                "Grounding": round(s["evidence_grounding"], 2),
                "Conflict": round(s["conflict_handling"], 2),
                "Overall": round(s["overall_score"], 3),
                "Failures": ", ".join(s["failure_labels"]) or "—",
            })

        import pandas as pd
        df = pd.DataFrame(rows)

        def color_overall(val):
            color = "#2d6a4f" if val >= 0.7 else ("#7a4f00" if val >= 0.4 else "#6b2737")
            return f"background-color: {color}"

        st.dataframe(
            df.style.map(color_overall, subset=["Overall"]),
            use_container_width=True,
            height=500,
        )


# ─── Page: Task Browser ──────────────────────────────────────────────────────
elif page == "📋 Task Browser":
    st.header("Benchmark Task Browser")

    tasks = load_tasks()
    summary = task_summary(tasks)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tasks", summary["total"])
    col2.metric("Domains", len(summary["domains"]))
    col3.metric("Evidence Statuses", len(summary["evidence_statuses"]))

    st.divider()

    # Filters
    domain_filter = st.multiselect(
        "Filter by domain",
        options=list(summary["domains"].keys()),
        default=[],
    )
    action_filter = st.multiselect(
        "Filter by gold action",
        options=list(summary["correct_actions"].keys()),
        default=[],
    )

    filtered = [
        t for t in tasks
        if (not domain_filter or t.domain in domain_filter)
        and (not action_filter or t.correct_action.value in action_filter)
    ]

    for task in filtered:
        status_class = f"tag-{task.evidence_status.value}"
        with st.expander(f"**{task.task_id}** — {task.domain} — {task.question[:70]}..."):
            st.markdown(f"**Scenario:** {task.scenario}")
            st.markdown(
                f"Evidence status: <span class='{status_class}'>{task.evidence_status.value}</span> &nbsp;"
                f"Gold action: `{task.correct_action.value}`",
                unsafe_allow_html=True,
            )
            st.markdown("**Evidence snippets:**")
            for i, snippet in enumerate(task.evidence):
                st.markdown(f"- `[{i}]` {snippet}")
            if task.gold_answer:
                st.markdown(f"**Gold answer:** {task.gold_answer}")
            st.markdown(f"**Gold reasoning:** *{task.gold_reasoning}*")
            if task.expected_failure_labels:
                labels_html = " ".join(
                    f'<span class="failure-chip">{f.value}</span>'
                    for f in task.expected_failure_labels
                )
                st.markdown(f"**Expected failure modes:** {labels_html}", unsafe_allow_html=True)


# ─── Page: Inspect Example ───────────────────────────────────────────────────
elif page == "🔍 Inspect Example":
    st.header("Inspect a Single Example")
    _, results = get_report_and_results()

    if not results:
        st.info("Run an evaluation first to inspect examples.")
        st.stop()

    # Select task
    task_ids = [r["task_id"] for r in results]
    selected_id = st.selectbox("Select task", task_ids)
    result = next(r for r in results if r["task_id"] == selected_id)

    tasks = load_tasks()
    task = next(t for t in tasks if t.task_id == selected_id)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Task")
        st.markdown(f"**Domain:** {task.domain}")
        st.markdown(f"**Question:** {task.question}")
        st.markdown(f"**Scenario:** {task.scenario}")
        st.markdown(f"**Evidence status:** `{task.evidence_status.value}`")
        st.markdown("**Evidence:**")
        for i, e in enumerate(task.evidence):
            cited = i in result["cited_evidence_indices"]
            icon = "✅" if cited else "○"
            st.markdown(f"{icon} `[{i}]` {e}")
        st.markdown(f"**Gold action:** `{task.correct_action.value}`")
        if task.gold_answer:
            st.markdown(f"**Gold answer:** {task.gold_answer}")

    with col2:
        st.subheader("Model Output")
        s = result["score"]
        st.markdown(f"**Predicted action:** `{result['predicted_action']}`")
        if result["answer_text"]:
            st.markdown(f"**Answer:** {result['answer_text']}")
        if result["reasoning_trace"]:
            st.markdown(f"**Reasoning:** *{result['reasoning_trace']}*")

        st.divider()
        st.subheader("Scores")
        score_cols = st.columns(3)
        score_cols[0].metric("Decision", f"{s['decision_quality']:.2f}")
        score_cols[1].metric("Grounding", f"{s['evidence_grounding']:.2f}")
        score_cols[2].metric("Conflict", f"{s['conflict_handling']:.2f}")
        st.metric("Overall Score", f"{s['overall_score']:.3f}")

        if s["failure_labels"]:
            st.markdown("**Failure labels:**")
            for label in s["failure_labels"]:
                st.markdown(
                    f'<span class="failure-chip">{label}</span>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("No failure labels detected")
