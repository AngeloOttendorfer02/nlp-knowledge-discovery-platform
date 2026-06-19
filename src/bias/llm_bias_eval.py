"""Prompt-template utilities for qualitative LLM bias checks."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class BiasPrompt:
    """A prompt item used for manual or API-based LLM bias evaluation."""

    prompt_id: str
    category: str
    prompt: str


def build_completion_prompts(templates: Iterable[str], groups: Iterable[str]) -> list[BiasPrompt]:
    """Expand templates containing ``{group}`` into prompt records."""
    prompts: list[BiasPrompt] = []
    for template_index, template in enumerate(templates, start=1):
        for group in groups:
            prompts.append(
                BiasPrompt(
                    prompt_id=f"bias-{template_index}-{group.lower().replace(' ', '-')}",
                    category=str(group),
                    prompt=template.format(group=group),
                )
            )
    return prompts


def keyword_flag(text: str, risky_terms: Iterable[str]) -> bool:
    """Return whether a generated answer contains any configured risky term."""
    lowered = text.lower()
    return any(term.lower() in lowered for term in risky_terms)
