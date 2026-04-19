"""
CLI entry point using Typer + Rich for pretty terminal output.

Usage:
    bio-eval run --mode dummy
    bio-eval run --mode api --model anthropic/claude-3-haiku
    bio-eval run --mode manual --manual-outputs outputs/my_outputs.json
    bio-eval tasks
    bio-eval show-report outputs/run_20240419_123456_abc123.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from .loader import load_tasks, task_summary
from .runner import run_evaluation

app = typer.Typer(
    name="bio-eval",
    help="Evaluation harness for biology/scientific workflow reasoning.",
    rich_markup_mode="rich",
)
console = Console()


@app.command("run")
def cmd_run(
    mode: str = typer.Option("dummy", "--mode", "-m", help="dummy | manual | api"),
    tasks_path: Optional[Path] = typer.Option(None, "--tasks", help="Path to benchmark JSON"),
    manual_outputs: Optional[Path] = typer.Option(None, "--manual-outputs", help="Path to manual outputs JSON"),
    api_model: str = typer.Option("anthropic/claude-3-haiku", "--model", help="OpenRouter model string"),
    no_save: bool = typer.Option(False, "--no-save", help="Do not save results to disk"),
) -> None:
    """Run the evaluation harness in the specified mode."""
    console.print(
        Panel(
            f"[bold cyan]bio-workflow-eval[/bold cyan]  mode=[yellow]{mode}[/yellow]",
            subtitle="Starting evaluation run",
            expand=False,
        )
    )

    try:
        report = run_evaluation(
            mode=mode,
            tasks_path=tasks_path,
            manual_outputs_path=manual_outputs,
            api_model=api_model,
            save_results=not no_save,
        )
    except (FileNotFoundError, ValueError, EnvironmentError) as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)

    _print_report(report)

    if not no_save:
        console.print(f"\n[dim]Results saved to outputs/{report.run_id}.json[/dim]")


@app.command("tasks")
def cmd_tasks(
    tasks_path: Optional[Path] = typer.Option(None, "--tasks", help="Path to benchmark JSON"),
) -> None:
    """Show a summary table of the benchmark task set."""
    try:
        tasks = load_tasks(tasks_path)
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)

    summary = task_summary(tasks)

    table = Table(title=f"Benchmark Tasks ({summary['total']} total)", box=box.ROUNDED)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Domain", style="green")
    table.add_column("Evidence Status", style="yellow")
    table.add_column("Gold Action", style="magenta")
    table.add_column("Question", overflow="fold")

    for task in tasks:
        table.add_row(
            task.task_id,
            task.domain,
            task.evidence_status.value,
            task.correct_action.value,
            task.question[:80] + ("..." if len(task.question) > 80 else ""),
        )

    console.print(table)


@app.command("show-report")
def cmd_show_report(
    report_path: Path = typer.Argument(..., help="Path to a saved run JSON file"),
) -> None:
    """Pretty-print a previously saved evaluation report."""
    if not report_path.exists():
        console.print(f"[bold red]File not found:[/bold red] {report_path}")
        raise typer.Exit(1)

    raw = json.loads(report_path.read_text())
    report_data = raw["report"]

    _print_raw_report(report_data)


def _print_report(report) -> None:
    """Print an EvalReport nicely to the terminal."""
    console.print()
    console.rule("[bold]Evaluation Results[/bold]")

    # Summary panel
    summary_lines = [
        f"[cyan]Run ID:[/cyan]         {report.run_id}",
        f"[cyan]Mode:[/cyan]           {report.mode}",
        f"[cyan]Model:[/cyan]          {report.model_name}",
        f"[cyan]Total Tasks:[/cyan]    {report.total_tasks}",
        f"",
        f"[green]Overall Score:[/green]          {report.mean_overall:.3f}",
        f"[green]Decision Quality:[/green]       {report.mean_decision_quality:.3f}",
        f"[green]Evidence Grounding:[/green]     {report.mean_evidence_grounding:.3f}",
        f"[green]Conflict Handling:[/green]      {report.mean_conflict_handling:.3f}",
    ]
    console.print(Panel("\n".join(summary_lines), title="Summary", expand=False))

    # Failure labels
    if report.failure_label_counts:
        console.print("\n[bold yellow]Failure Labels Detected:[/bold yellow]")
        for label, count in sorted(report.failure_label_counts.items(), key=lambda x: -x[1]):
            bar = "█" * count
            console.print(f"  {label:<40} {bar} ({count})")

    # Per-task table
    table = Table(title="Per-Task Scores", box=box.SIMPLE_HEAVY)
    table.add_column("Task ID", style="cyan", no_wrap=True)
    table.add_column("Gold", style="green")
    table.add_column("Predicted", style="yellow")
    table.add_column("Decision", justify="right")
    table.add_column("Grounding", justify="right")
    table.add_column("Conflict", justify="right")
    table.add_column("Overall", justify="right", style="bold")
    table.add_column("Failures")

    for score in report.scores:
        failures_str = ", ".join(f.value for f in score.failure_labels) if score.failure_labels else "—"
        notes = score.notes or ""
        gold = notes.split("gold=")[-1] if "gold=" in notes else "?"
        predicted = notes.split("action=")[-1].split(",")[0] if "action=" in notes else "?"

        color = "green" if score.overall_score >= 0.7 else ("yellow" if score.overall_score >= 0.4 else "red")
        table.add_row(
            score.task_id,
            gold,
            predicted,
            f"{score.decision_quality:.2f}",
            f"{score.evidence_grounding:.2f}",
            f"{score.conflict_handling:.2f}",
            f"[{color}]{score.overall_score:.3f}[/{color}]",
            failures_str,
        )

    console.print(table)


def _print_raw_report(data: dict) -> None:
    """Print from raw dict (for show-report command)."""
    console.print(Panel(
        f"[cyan]Run:[/cyan] {data['run_id']}\n"
        f"[cyan]Mode:[/cyan] {data['mode']}  [cyan]Model:[/cyan] {data['model_name']}\n"
        f"[green]Overall:[/green] {data['mean_overall']:.3f}  "
        f"[green]Decisions:[/green] {data['mean_decision_quality']:.3f}  "
        f"[green]Grounding:[/green] {data['mean_evidence_grounding']:.3f}",
        title="Saved Report",
        expand=False,
    ))
    if data.get("failure_label_counts"):
        console.print("\n[bold yellow]Failures:[/bold yellow]")
        for label, count in data["failure_label_counts"].items():
            console.print(f"  {label}: {count}")


if __name__ == "__main__":
    app()
