#!/usr/bin/env python3
"""Prompt Token Diff Optimizer — CLI entry point.

Usage:
  python optimizer.py --prompt prompt.txt --evals evals.json
  python optimizer.py --prompt prompt.txt --evals evals.json --metric llm --granularity headers
  python optimizer.py --prompt prompt.txt --evals evals.json --export results.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import openai
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

load_dotenv()

from ablator import parse_sections, generate_ablations
from evaluator import evaluate_baseline, evaluate_variants, count_tokens
from reporter import render_table, export_csv

console = Console()


@click.command()
@click.option("--prompt", required=True, type=click.Path(exists=True), help="Path to the system prompt file.")
@click.option("--evals", required=True, type=click.Path(exists=True), help="Path to the eval set JSON file.")
@click.option(
    "--metric",
    default="llm",
    type=click.Choice(["exact", "contains", "llm"], case_sensitive=False),
    show_default=True,
    help="Scoring metric. 'llm' uses gpt-4o-mini as a judge.",
)
@click.option(
    "--granularity",
    default="auto",
    type=click.Choice(["auto", "headers", "dividers", "paragraphs", "sentences"], case_sensitive=False),
    show_default=True,
    help="How to split the prompt into sections.",
)
@click.option("--model", default="gpt-4o-mini", show_default=True, help="Model to test against.")
@click.option("--max-tokens", default=512, show_default=True, help="Max tokens for each model response.")
@click.option("--export", default=None, type=click.Path(), help="Export results to a CSV file.")
@click.option(
    "--safe-threshold",
    default=0.02,
    show_default=True,
    help="Quality delta at or above this (negative) value is 'safe to cut'.",
)
@click.option(
    "--warn-threshold",
    default=0.05,
    show_default=True,
    help="Quality delta below safe but above this threshold is 'borderline'.",
)
@click.option("--delay", default=0.3, show_default=True, help="Seconds between API calls (rate-limit buffer).")
def main(
    prompt: str,
    evals: str,
    metric: str,
    granularity: str,
    model: str,
    max_tokens: int,
    export: str | None,
    safe_threshold: float,
    warn_threshold: float,
    delay: float,
) -> None:
    prompt_text = Path(prompt).read_text()
    eval_set = json.loads(Path(evals).read_text())

    if not isinstance(eval_set, list) or not eval_set:
        console.print("[red]Error:[/red] eval set must be a non-empty JSON array.")
        sys.exit(1)

    for item in eval_set:
        if "input" not in item or "expected" not in item:
            console.print('[red]Error:[/red] each eval item must have "input" and "expected" keys.')
            sys.exit(1)

    client = openai.OpenAI()

    console.print(f"\n[bold cyan]Prompt Token Diff Optimizer[/bold cyan]")
    console.print(f"  Prompt: {prompt}  |  Evals: {len(eval_set)} items  |  Metric: {metric}  |  Model: {model}\n")

    sections = parse_sections(prompt_text, granularity=granularity)
    console.print(f"Detected [bold]{len(sections)}[/bold] sections (granularity: {granularity})")
    for s in sections:
        console.print(f"  • {s.label}")
    console.print()

    variants = generate_ablations(prompt_text, sections)
    full_tokens = count_tokens(prompt_text, model)
    console.print(f"Full prompt: [bold]{full_tokens:,}[/bold] tokens\n")

    # Baseline
    console.print("[bold]Step 1/2:[/bold] Measuring baseline quality (full prompt)…")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task(f"Running {len(eval_set)} eval items…", total=None)
        baseline = evaluate_baseline(client, prompt_text, eval_set, metric, model, max_tokens, delay)
        progress.remove_task(task)
    console.print(f"  Baseline quality: [bold]{baseline:.4f}[/bold]\n")

    # Ablations
    console.print(f"[bold]Step 2/2:[/bold] Ablating {len(variants)} sections…")
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        ptask = progress.add_task("Ablating…", total=len(variants))

        def on_progress(i: int, total: int, label: str) -> None:
            progress.update(ptask, completed=i, description=f"[{i+1}/{total}] {label[:50]}")

        results = evaluate_variants(
            client=client,
            full_prompt=prompt_text,
            variants=variants,
            eval_set=eval_set,
            metric=metric,
            model=model,
            max_tokens=max_tokens,
            baseline_quality=baseline,
            rate_limit_delay=delay,
            progress_callback=on_progress,
        )
        progress.update(ptask, completed=len(variants), description="Done.")

    console.print()
    render_table(results, baseline, full_tokens, safe_threshold, warn_threshold)

    if export:
        export_csv(results, export)


if __name__ == "__main__":
    main()
