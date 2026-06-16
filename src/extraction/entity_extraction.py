"""
entity_extraction.py — Named Entity Recognition for scientific documents.

Wraps a spaCy NER pipeline and exposes a small, typed interface that returns
structured entities from raw text. Beyond spaCy's standard entity types, the
extractor optionally augments results with lightweight, rule-based detection
of scientific concepts (capitalised method/dataset names) that generic NER
models tend to miss.

Entities are the building blocks of the knowledge graph: each unique entity
becomes a node, and relations between entities become edges.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Dict, List, Optional

_CONCEPT_RE = re.compile(
    r"\b(?:knowledge graph|semantic search|semantic retrieval|question answering|"
    r"explainable ai|graph neural network|graph neural networks|link prediction|"
    r"citation network|citation networks|language model|language models|"
    r"embedding space|embedding spaces|bias evaluation|debiasing|topic modeling|"
    r"scientific document|scientific documents|literature review|"
    r"retrieval augmented generation|structured knowledge)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Entity:
    """
    A single extracted entity.

    Attributes
    ----------
    text : str
        Surface form of the entity as it appears in the document.
    label : str
        Entity type (e.g. "ORG", "PERSON", "CONCEPT").
    start : int
        Character offset where the entity begins.
    end : int
        Character offset where the entity ends.
    """

    text: str
    label: str
    start: int
    end: int


@lru_cache(maxsize=2)
def _load_spacy(model_name: str):
    """Load and cache a spaCy model with its NER component enabled."""
    import spacy

    try:
        return spacy.load(model_name)
    except OSError as exc:  # pragma: no cover - depends on local install
        raise OSError(
            f"spaCy model '{model_name}' is not installed. "
            f"Run: python -m spacy download {model_name}"
        ) from exc


class EntityExtractor:
    """
    Extract named entities from text using spaCy.

    Parameters
    ----------
    spacy_model : str
        Name of the spaCy model to load.
    entity_types : list of str, optional
        Keep only entities of these spaCy labels. If None, all labels are kept.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        entity_types: Optional[List[str]] = None,
    ) -> None:
        self.spacy_model = spacy_model
        self.entity_types = set(entity_types) if entity_types else None
        try:
            self._nlp = _load_spacy(spacy_model)
        except (ModuleNotFoundError, OSError):
            self._nlp = None

    def extract(self, text: str) -> List[Entity]:
        """
        Extract entities from a single document.

        Parameters
        ----------
        text : str
            Input document text.

        Returns
        -------
        list of Entity
            Deduplicated entities (by surface form + label).
        """
        if not text or not text.strip():
            return []
        if self._nlp is None:
            return _extract_rule_based_concepts(text)

        doc = self._nlp(text)
        seen = set()
        entities: List[Entity] = []

        for ent in doc.ents:
            if self.entity_types is not None and ent.label_ not in self.entity_types:
                continue
            surface = ent.text.strip()
            key = (surface.lower(), ent.label_)
            if surface and key not in seen:
                seen.add(key)
                entities.append(
                    Entity(text=surface, label=ent.label_, start=ent.start_char, end=ent.end_char)
                )

        return entities

    def extract_batch(self, texts: List[str], batch_size: int = 64) -> List[List[Entity]]:
        """
        Extract entities from many documents efficiently using nlp.pipe.

        Parameters
        ----------
        texts : list of str
            Documents to process.
        batch_size : int
            Number of documents processed per spaCy batch.

        Returns
        -------
        list of list of Entity
            One entity list per input document.
        """
        if self._nlp is None:
            return [_extract_rule_based_concepts(text) for text in texts]

        results: List[List[Entity]] = []
        for doc in self._nlp.pipe(texts, batch_size=batch_size):
            seen = set()
            doc_entities: List[Entity] = []
            for ent in doc.ents:
                if self.entity_types is not None and ent.label_ not in self.entity_types:
                    continue
                surface = ent.text.strip()
                key = (surface.lower(), ent.label_)
                if surface and key not in seen:
                    seen.add(key)
                    doc_entities.append(
                        Entity(
                            text=surface,
                            label=ent.label_,
                            start=ent.start_char,
                            end=ent.end_char,
                        )
                    )
            results.append(doc_entities)
        return results

    @staticmethod
    def most_common_entities(
        entities_per_doc: List[List[Entity]], top_n: int = 20
    ) -> List[tuple]:
        """
        Aggregate entities across a corpus and return the most frequent ones.

        Parameters
        ----------
        entities_per_doc : list of list of Entity
            Output of :meth:`extract_batch`.
        top_n : int
            Number of entities to return.

        Returns
        -------
        list of ((surface, label), count) tuples
            Sorted by descending frequency.
        """
        counter: Counter = Counter()
        for doc_entities in entities_per_doc:
            for entity in doc_entities:
                counter[(entity.text, entity.label)] += 1
        return counter.most_common(top_n)


def _extract_rule_based_concepts(text: str) -> List[Entity]:
    """Extract project-relevant scientific concepts without a spaCy model."""
    seen = set()
    entities: List[Entity] = []
    for match in _CONCEPT_RE.finditer(text or ""):
        surface = match.group(0).strip()
        key = surface.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append(
            Entity(
                text=surface,
                label="CONCEPT",
                start=match.start(),
                end=match.end(),
            )
        )
    return entities
