"""Utilities for lightweight bias analysis in embedding spaces."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np


@dataclass(frozen=True)
class AssociationScore:
    """Association of one target group with two attribute sets."""

    target: str
    mean_attribute_a: float
    mean_attribute_b: float
    differential: float


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity for two vectors, safely handling zero vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def mean_vector(words: Sequence[str], embeddings: Mapping[str, np.ndarray]) -> np.ndarray:
    """Average all available word vectors from ``embeddings``."""
    vectors = [np.asarray(embeddings[word], dtype=float) for word in words if word in embeddings]
    if not vectors:
        raise ValueError("None of the requested words are present in the embedding map")
    return np.mean(vectors, axis=0)


def association(
    target_words: Sequence[str],
    attribute_a: Sequence[str],
    attribute_b: Sequence[str],
    embeddings: Mapping[str, np.ndarray],
) -> list[AssociationScore]:
    """Compute per-target association difference against two attribute sets."""
    attr_a = [word for word in attribute_a if word in embeddings]
    attr_b = [word for word in attribute_b if word in embeddings]
    if not attr_a or not attr_b:
        raise ValueError("Both attribute sets need at least one available vector")

    scores: list[AssociationScore] = []
    for target in target_words:
        if target not in embeddings:
            continue
        vec = embeddings[target]
        mean_a = float(np.mean([cosine_similarity(vec, embeddings[w]) for w in attr_a]))
        mean_b = float(np.mean([cosine_similarity(vec, embeddings[w]) for w in attr_b]))
        scores.append(AssociationScore(target, mean_a, mean_b, mean_a - mean_b))
    return scores


def group_bias_score(
    target_a: Sequence[str],
    target_b: Sequence[str],
    attribute_a: Sequence[str],
    attribute_b: Sequence[str],
    embeddings: Mapping[str, np.ndarray],
) -> float:
    """Return a simple WEAT-style differential association score."""
    assoc_a = association(target_a, attribute_a, attribute_b, embeddings)
    assoc_b = association(target_b, attribute_a, attribute_b, embeddings)
    if not assoc_a or not assoc_b:
        return 0.0
    return float(np.mean([s.differential for s in assoc_a]) - np.mean([s.differential for s in assoc_b]))
