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


# Capitalisation-based patterns. These MUST be matched case-sensitively, so they
# only fire on genuinely capitalised tokens (proper nouns / acronyms). Matching
# them case-insensitively would wrongly capture ordinary words such as "uses",
# "for", or "research".
_CASE_SENSITIVE_PATTERNS = [
    # A run of capitalised words, e.g. "Graph Neural Networks"
    r"\b[A-Z][A-Za-z0-9]+(?:[ -][A-Z][A-Za-z0-9]+){0,4}\b",
    # Acronyms / model names, e.g. "BERT", "FAISS", "GNN", "LSTM-CRF"
    r"\b[A-Z]{2,}[A-Za-z0-9-]*\b",
]

# Known scientific phrases. These are safe to match case-insensitively because
# the full phrase is specific and cannot collide with ordinary stopwords.
_PHRASE_PATTERN = (
    r"\b(?:graph neural networks?|knowledge graphs?|language models?|"
    r"semantic retrieval|topic modeling|named entity recognition|"
    r"relation extraction|retrieval augmented generation|"
    r"transformer models?|citation networks?)\b"
)

# Common words that should never stand alone as an entity and should be trimmed
# from the edges of a multi-word match (e.g. "Our BERT" -> "BERT").
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "but", "by", "for", "from", "in", "is",
    "it", "its", "of", "on", "or", "our", "that", "the", "their", "these",
    "this", "those", "to", "use", "uses", "using", "we", "with", "within",
    "show", "shows", "propose", "proposes", "research", "results", "method",
    "methods", "model", "models", "paper", "papers",
}


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


def _trim_stopwords(surface: str) -> str:
    """Remove leading and trailing stopwords from a multi-word match.

    Turns spans like "We propose Graph Neural Networks" into
    "Graph Neural Networks" and "Our BERT" into "BERT".
    """
    words = surface.split()
    while words and words[0].lower() in _STOPWORDS:
        words.pop(0)
    while words and words[-1].lower() in _STOPWORDS:
        words.pop()
    return " ".join(words)


def _rule_based_entities(text: str, entity_types: Optional[set[str]] = None) -> List[Entity]:
    """Extract useful fallback entities from scientific text without spaCy NER.

    Capitalisation-based patterns are matched case-sensitively so ordinary words
    (``uses``, ``for``, ``research``) are not mistaken for entities; known
    scientific phrases are matched case-insensitively. Leading/trailing
    stopwords are trimmed and single stopwords are discarded.
    """
    if entity_types is not None and "CONCEPT" not in entity_types:
        return []

    seen: set[str] = set()
    entities: List[Entity] = []

    # (pattern, regex flags) — caps patterns are case-sensitive, phrases are not
    pattern_specs = [(p, 0) for p in _CASE_SENSITIVE_PATTERNS]
    pattern_specs.append((_PHRASE_PATTERN, re.IGNORECASE))

    for pattern, flags in pattern_specs:
        for match in re.finditer(pattern, text, flags=flags):
            raw = match.group(0)
            cleaned = raw.strip(" .,:;()[]{}")
            surface = _trim_stopwords(cleaned)

            # Discard empties, single characters, and lone stopwords
            if len(surface) < 3 or surface.lower() in _STOPWORDS:
                continue

            # Recompute offsets so start/end point at the TRIMMED surface, not the
            # original match span (e.g. "Our BERT" -> "BERT" must move start past
            # "Our "). The trimmed surface is always a literal substring of the
            # matched text, so locating it within the span yields exact offsets.
            local = raw.find(surface)
            if local >= 0:
                start = match.start() + local
                end = start + len(surface)
            else:  # pragma: no cover - defensive; surface is normally a substring
                start, end = match.start(), match.end()

            key = surface.lower()
            if key not in seen:
                seen.add(key)
                entities.append(Entity(surface, "CONCEPT", start, end))

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
