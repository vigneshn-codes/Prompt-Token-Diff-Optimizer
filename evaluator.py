"""Run prompt variants against an eval set and collect quality scores."""

from __future__ import annotations

import time
from dataclasses import dataclass

import openai
import tiktoken

from ablator import AblationVariant, Section
from scorer import aggregate, score


@dataclass
class EvalResult:
    section: Section
    token_count_full: int      # tokens in the full prompt
    token_count_ablated: int   # tokens in the prompt with this section removed
    tokens_saved: int
    quality_full: float        # baseline quality (full prompt)
    quality_ablated: float     # quality after removing this section
    quality_delta: float       # ablated - full  (negative = section helped)
    impact_per_token: float    # quality_delta / tokens_saved


def count_tokens(prompt: str, model: str) -> int:
    """Count tokens locally via tiktoken — no API call needed."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    # 4 tokens overhead for system message framing + 3 for reply primer
    return len(enc.encode(prompt)) + 7


def run_prompt(
    client: openai.OpenAI,
    system: str,
    user_input: str,
    model: str,
    max_tokens: int,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_input},
        ],
    )
    return resp.choices[0].message.content.strip()


def evaluate_baseline(
    client: openai.OpenAI,
    full_prompt: str,
    eval_set: list[dict],
    metric: str,
    model: str,
    max_tokens: int,
    rate_limit_delay: float,
) -> float:
    """Compute quality of the full (unmodified) prompt."""
    scores, weights = [], []
    for item in eval_set:
        output = run_prompt(client, full_prompt, item["input"], model, max_tokens)
        s = score(output, item["expected"], metric, item["input"])
        scores.append(s)
        weights.append(item.get("weight", 1.0))
        time.sleep(rate_limit_delay)
    return aggregate(scores, weights)


def evaluate_variants(
    client: openai.OpenAI,
    full_prompt: str,
    variants: list[AblationVariant],
    eval_set: list[dict],
    metric: str,
    model: str,
    max_tokens: int,
    baseline_quality: float,
    rate_limit_delay: float,
    progress_callback=None,
) -> list[EvalResult]:
    full_tokens = count_tokens(full_prompt, model)

    results: list[EvalResult] = []
    for i, variant in enumerate(variants):
        if progress_callback:
            progress_callback(i, len(variants), variant.removed_section.label)

        ablated_tokens = count_tokens(variant.ablated_prompt, model)
        tokens_saved = full_tokens - ablated_tokens

        scores, weights = [], []
        for item in eval_set:
            output = run_prompt(
                client, variant.ablated_prompt, item["input"], model, max_tokens
            )
            s = score(output, item["expected"], metric, item["input"])
            scores.append(s)
            weights.append(item.get("weight", 1.0))
            time.sleep(rate_limit_delay)

        q_ablated = aggregate(scores, weights)
        q_delta = q_ablated - baseline_quality
        impact = q_delta / tokens_saved if tokens_saved > 0 else 0.0

        results.append(
            EvalResult(
                section=variant.removed_section,
                token_count_full=full_tokens,
                token_count_ablated=ablated_tokens,
                tokens_saved=tokens_saved,
                quality_full=baseline_quality,
                quality_ablated=q_ablated,
                quality_delta=q_delta,
                impact_per_token=impact,
            )
        )

    return results
