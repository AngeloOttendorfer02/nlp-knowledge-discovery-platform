"""
entity_extraction.py — Named Entity Recognition for scientific documents.

The extractor uses spaCy when the configured model is available. If the model is
missing, it emits a clear warning and falls back to a lightweight rule-based
scientific term extractor instead of silently returning no entities.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional


@dataclass(frozen=True)
class Entity:
    """A single extracted entity."""

    text: str
    label: str
    start: int
    end: int


_SCIENTIFIC_PATTERNS = [
    r"\b[A-Z][A-Za-z0-9]+(?:[- ][A-Z]?[A-Za-z0-9]+){0,4}\b",
    r"\b(?:[A-Z]{2,}[A-Za-z0-9-]*)\b",
    r"\b(?:graph neural networks?|knowledge graphs?|language models?|semantic retrieval|topic modeling|named entity recognition|relation extraction|retrieval augmented generation|transformer models?|citation networks?)\b",
]


@lru_cache(maxsize=2)
def _load_spacy(model_name: str):
    """Load and cache a spaCy model, falling back with an explicit warning."""
    import spacy

    try:
        nlp = spacy.load(model_name)
        nlp.meta["project_fallback"] = False
        return nlp
    except OSError:  # pragma: no cover - depends on local install
        warnings.warn(
            f"spaCy model '{model_name}' is not installed. Falling back to a blank "
            "English pipeline plus rule-based scientific entity extraction. "
            f"Install it with: python -m spacy download {model_name}",
            RuntimeWarning,
            stacklevel=2,
        )
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        nlp.meta["project_fallback"] = True
        return nlp


def _rule_based_entities(text: str, entity_types: Optional[set[str]] = None) -> List[Entity]:
    """Extract useful fallback entities from scientific text without spaCy NER."""
    if entity_types is not None and "CONCEPT" not in entity_types:
        return []

    seen: set[str] = set()
    entities: list[Entity] = []

    for pattern in _SCIENTIFIC_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            surface = re.sub(r"\s+", " ", match.group(0)).strip(" .,:;()[]{}")
            if len(surface) < 3:
                continue
            if surface.lower() in {"the", "and", "with", "this", "that", "using"}:
                continue
            key = surface.lower()
            if key not in seen:
                seen.add(key)
                entities.append(Entity(surface, "CONCEPT", match.start(), match.end()))

    return entities


class EntityExtractor:
    """Extract named entities from text using spaCy with a rule-based fallback."""

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        entity_types: Optional[List[str]] = None,
    ) -> None:
        self.spacy_model = spacy_model
        self.entity_types = set(entity_types) if entity_types else None
        self._nlp = _load_spacy(spacy_model)
        self.uses_fallback = bool(self._nlp.meta.get("project_fallback", False))
        self.has_ner = "ner" in self._nlp.pipe_names

        if not self.has_ner:
            warnings.warn(
                "The active spaCy pipeline has no NER component. Entity extraction "
                "will use the project's rule-based scientific fallback.",
                RuntimeWarning,
                stacklevel=2,
            )


    def _entities_from_doc(self, doc) -> List[Entity]:
        seen = set()
        entities: List[Entity] = []
        for ent in getattr(doc, "ents", []):
            if self.entity_types is not None and ent.label_ not in self.entity_types:
                continue
            surface = ent.text.strip()
            key = (surface.lower(), ent.label_)
            if surface and key not in seen:
                seen.add(key)
                entities.append(Entity(text=surface, label=ent.label_, start=ent.start_char, end=ent.end_char))
        return entities

    def extract(self, text: str) -> List[Entity]:
        if not text or not text.strip():
            return []

        try:
            doc = self._nlp(text)
            entities = self._entities_from_doc(doc)
        except Exception:
            entities = []

        if not entities:
            entities = _rule_based_entities(text, self.entity_types)

        return entities

    def extract_batch(self, texts: List[str], batch_size: int = 64) -> List[List[Entity]]:
        results: List[List[Entity]] = []
        try:
            docs = self._nlp.pipe(texts, batch_size=batch_size)
        except Exception:
            return [self.extract(str(text)) for text in texts]

        for text, doc in zip(texts, docs):
            doc_entities = self._entities_from_doc(doc)
            if not doc_entities:
                doc_entities = _rule_based_entities(str(text), self.entity_types)
            results.append(doc_entities)
        return results

    @staticmethod
    def most_common_entities(entities_per_doc: List[List[Entity]], top_n: int = 20) -> List[tuple]:
        counter: Counter = Counter()
        for doc_entities in entities_per_doc:
            for entity in doc_entities:
                counter[(entity.text, entity.label)] += 1
        return counter.most_common(top_n)
