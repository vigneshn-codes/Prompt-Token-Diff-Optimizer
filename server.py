"""FastAPI backend for Prompt Token Diff Optimizer.

Streams ablation results via Server-Sent Events so the frontend
can show live progress.

Run:
  uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import json
import sys
import asyncio
from pathlib import Path
from typing import AsyncGenerator

import openai
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

# Ensure local modules are importable
sys.path.insert(0, str(Path(__file__).parent))

from ablator import parse_sections, generate_ablations
from evaluator import count_tokens, run_prompt, evaluate_baseline
from scorer import aggregate, score

app = FastAPI(title="Prompt Token Diff Optimizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class AnalyzeRequest(BaseModel):
    prompt: str
    evals: list[dict]
    metric: str = "llm"
    granularity: str = "auto"
    model: str = "gpt-4o-mini"
    max_tokens: int = 512
    safe_threshold: float = 0.02
    warn_threshold: float = 0.05
    delay: float = 0.3


def _sse(event_type: str, data: dict) -> str:
    payload = json.dumps({"type": event_type, **data})
    return f"data: {payload}\n\n"


def _recommendation(delta: float, saved: int, safe: float, warn: float) -> str:
    if saved <= 0:
        return "no_savings"
    if delta >= -safe:
        return "cut"
    if delta >= -warn:
        return "borderline"
    return "keep"


async def stream_analysis(req: AnalyzeRequest) -> AsyncGenerator[str, None]:
    client = openai.OpenAI()

    # Parse sections
    try:
        sections = parse_sections(req.prompt, granularity=req.granularity)
    except Exception as e:
        yield _sse("error", {"message": f"Failed to parse prompt: {e}"})
        return

    if not sections:
        yield _sse("error", {"message": "No sections detected in prompt."})
        return

    variants = generate_ablations(req.prompt, sections)

    # Token counting is local (tiktoken), no need to thread
    try:
        full_tokens = count_tokens(req.prompt, req.model)
    except Exception as e:
        yield _sse("error", {"message": f"Token count failed: {e}"})
        return

    yield _sse("init", {
        "section_count": len(sections),
        "full_tokens": full_tokens,
        "sections": [{"label": s.label, "char_count": len(s.text)} for s in sections],
    })

    # Baseline
    yield _sse("progress", {"phase": "baseline", "message": "Measuring baseline quality…"})
    try:
        baseline = await asyncio.to_thread(
            evaluate_baseline,
            client, req.prompt, req.evals, req.metric, req.model, req.max_tokens, req.delay,
        )
    except Exception as e:
        yield _sse("error", {"message": f"Baseline evaluation failed: {e}"})
        return

    yield _sse("baseline", {"quality": baseline, "full_tokens": full_tokens})

    # Ablations
    results = []
    for i, variant in enumerate(variants):
        section = variant.removed_section
        yield _sse("progress", {
            "phase": "ablation",
            "current": i + 1,
            "total": len(variants),
            "section_label": section.label,
        })

        try:
            ablated_tokens = count_tokens(variant.ablated_prompt, req.model)
            tokens_saved = full_tokens - ablated_tokens

            scores, weights = [], []
            for item in req.evals:
                output = await asyncio.to_thread(
                    run_prompt, client, variant.ablated_prompt,
                    item["input"], req.model, req.max_tokens,
                )
                s = score(output, item["expected"], req.metric, item["input"])
                scores.append(s)
                weights.append(item.get("weight", 1.0))
                await asyncio.sleep(req.delay)

            q_ablated = aggregate(scores, weights)
            q_delta = q_ablated - baseline
            impact = q_delta / tokens_saved if tokens_saved > 0 else 0.0
            rec = _recommendation(q_delta, tokens_saved, req.safe_threshold, req.warn_threshold)

            row = {
                "section_label": section.label,
                "tokens_saved": tokens_saved,
                "quality_ablated": round(q_ablated, 4),
                "quality_delta": round(q_delta, 4),
                "impact_per_token": round(impact, 6),
                "recommendation": rec,
            }
            results.append(row)
            yield _sse("row", row)

        except Exception as e:
            yield _sse("row_error", {"section_label": section.label, "message": str(e)})

    total_saveable = sum(r["tokens_saved"] for r in results if r["recommendation"] == "cut")
    yield _sse("done", {
        "results": results,
        "total_saveable_tokens": total_saveable,
        "saveable_pct": round(total_saveable / full_tokens * 100, 1) if full_tokens else 0,
    })


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    return StreamingResponse(
        stream_analysis(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/", response_class=HTMLResponse)
async def root():
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return HTMLResponse(html_file.read_text())
    return HTMLResponse("<h1>Static files not found. Place index.html in ./static/</h1>")


@app.get("/health")
async def health():
    return {"status": "ok"}
