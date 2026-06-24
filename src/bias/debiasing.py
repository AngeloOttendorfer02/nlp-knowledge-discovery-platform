"""Simple embedding debiasing helpers used by the bias-analysis notebook."""

from __future__ import annotations
from typing import Mapping, MutableMapping, Sequence
import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return a unit vector, preserving zero vectors."""
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def bias_direction(positive_words: Sequence[str], negative_words: Sequence[str], embeddings: Mapping[str, np.ndarray]) -> np.ndarray:
    """Estimate a bias direction from paired or grouped word lists."""
    positives = [embeddings[w] for w in positive_words if w in embeddings]
    negatives = [embeddings[w] for w in negative_words if w in embeddings]
    if not positives or not negatives:
        raise ValueError("Both word sets need at least one available embedding")
    return normalize(np.mean(positives, axis=0) - np.mean(negatives, axis=0))


def neutralize(vector: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove the projection of ``vector`` on ``direction``."""
    direction = normalize(direction)
    vector = np.asarray(vector, dtype=float)
    return vector - np.dot(vector, direction) * direction


def neutralize_embeddings(
    words: Sequence[str],
    embeddings: MutableMapping[str, np.ndarray],
    direction: np.ndarray,
    *,
    inplace: bool = False,
) -> MutableMapping[str, np.ndarray]:
    """Neutralize selected embeddings and return the updated mapping."""
    output = embeddings if inplace else dict(embeddings)
    for word in words:
        if word in output:
            output[word] = neutralize(output[word], direction)
    return output
