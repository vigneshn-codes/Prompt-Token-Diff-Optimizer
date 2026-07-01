"""Scoring metrics for evaluating prompt output quality."""

from __future__ import annotations

import re
import openai

# Lazily created client (only for llm-judge mode)
_client: openai.OpenAI | None = None


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI()
    return _client


def score(
    output: str,
    expected: str,
    metric: str,
    input_text: str = "",
) -> float:
    """Return a quality score in [0.0, 1.0].

    metric options:
      "exact"    – 1.0 if output == expected (case-insensitive stripped)
      "contains" – 1.0 if expected appears anywhere in output
      "llm"      – LLM-as-judge rating 1–10 normalised to [0,1]
    """
    if metric == "exact":
        return 1.0 if output.strip().lower() == expected.strip().lower() else 0.0

    if metric == "contains":
        return 1.0 if expected.strip().lower() in output.strip().lower() else 0.0

    if metric == "llm":
        return _llm_judge(input_text, output, expected)

    raise ValueError(f"Unknown metric: {metric!r}. Choose from: exact, contains, llm")


def _llm_judge(input_text: str, output: str, expected: str) -> float:
    client = _get_client()
    judge_prompt = f"""You are an impartial evaluator. Rate the quality of an AI assistant's output on a scale from 1 to 10.

Input given to the assistant:
<input>{input_text}</input>

Expected/ideal output:
<expected>{expected}</expected>

Actual output produced:
<actual>{output}</actual>

Rate how well the actual output fulfils the intent of the expected output.
Respond with ONLY a single integer from 1 to 10. Nothing else."""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=8,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    raw = resp.choices[0].message.content.strip()
    m = re.search(r"\d+", raw)
    if m:
        return min(max(int(m.group()), 1), 10) / 10.0
    return 0.5  # fallback


def aggregate(scores: list[float], weights: list[float] | None = None) -> float:
    if not scores:
        return 0.0
    if weights is None:
        weights = [1.0] * len(scores)
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return sum(s * w for s, w in zip(scores, weights)) / total_weight
