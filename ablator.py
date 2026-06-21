"""Prompt parsing and ablation logic."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Section:
    index: int
    label: str
    text: str
    tokens: int = 0  # filled in by evaluator


@dataclass
class AblationVariant:
    removed_section: Section
    ablated_prompt: str


def parse_sections(prompt: str, granularity: str = "auto") -> list[Section]:
    """Split a prompt into labeled sections.

    granularity:
      "auto"      – try headers, then dividers, then paragraphs
      "headers"   – markdown ## / ### / # headers
      "dividers"  – --- SECTION: label --- style
      "paragraphs"– blank-line separated blocks
      "sentences" – individual sentences
    """
    if granularity == "auto":
        if re.search(r"^#{1,3} .+", prompt, re.MULTILINE):
            granularity = "headers"
        elif re.search(r"^---\s*.+\s*---", prompt, re.MULTILINE):
            granularity = "dividers"
        else:
            granularity = "paragraphs"

    if granularity == "headers":
        return _split_by_headers(prompt)
    if granularity == "dividers":
        return _split_by_dividers(prompt)
    if granularity == "sentences":
        return _split_by_sentences(prompt)
    return _split_by_paragraphs(prompt)


def _split_by_headers(prompt: str) -> list[Section]:
    pattern = re.compile(r"^(#{1,3} .+)$", re.MULTILINE)
    matches = list(pattern.finditer(prompt))
    if not matches:
        return _split_by_paragraphs(prompt)

    sections: list[Section] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
        text = prompt[start:end].strip()
        if text:
            sections.append(Section(index=i, label=m.group(1).strip(), text=text))

    # Capture any text before the first header as a preamble
    preamble = prompt[: matches[0].start()].strip()
    if preamble:
        sections.insert(0, Section(index=-1, label="[preamble]", text=preamble))

    return sections


def _split_by_dividers(prompt: str) -> list[Section]:
    pattern = re.compile(r"^---\s*(.+?)\s*---$", re.MULTILINE)
    matches = list(pattern.finditer(prompt))
    if not matches:
        return _split_by_paragraphs(prompt)

    sections: list[Section] = []
    for i, m in enumerate(matches):
        label = m.group(1).strip()
        content_start = m.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
        text = (m.group(0) + "\n" + prompt[content_start:content_end]).strip()
        if text:
            sections.append(Section(index=i, label=label, text=text))

    preamble = prompt[: matches[0].start()].strip()
    if preamble:
        sections.insert(0, Section(index=-1, label="[preamble]", text=preamble))

    return sections


def _split_by_paragraphs(prompt: str) -> list[Section]:
    blocks = re.split(r"\n{2,}", prompt.strip())
    sections = []
    for i, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue
        first_line = block.splitlines()[0][:60]
        label = f"[para {i+1}] {first_line}{'…' if len(block.splitlines()[0]) > 60 else ''}"
        sections.append(Section(index=i, label=label, text=block))
    return sections


def _split_by_sentences(prompt: str) -> list[Section]:
    # Naive sentence split that avoids splitting on abbreviations
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", prompt.strip())
    sections = []
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if len(sent) < 10:
            continue
        preview = sent[:60]
        label = f"[sent {i+1}] {preview}{'…' if len(sent) > 60 else ''}"
        sections.append(Section(index=i, label=label, text=sent))
    return sections


def generate_ablations(prompt: str, sections: list[Section]) -> list[AblationVariant]:
    """For each section, produce a version of the prompt with that section removed."""
    variants = []
    for section in sections:
        ablated = prompt.replace(section.text, "").strip()
        # Clean up double blank lines left behind
        ablated = re.sub(r"\n{3,}", "\n\n", ablated)
        variants.append(AblationVariant(removed_section=section, ablated_prompt=ablated))
    return variants
