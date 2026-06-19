"""
relation_extraction.py — Extract relationships between entities.

Relationships turn an unconnected bag of entities into a graph. This module
implements two strategies:

1. **Co-occurrence relations** — two entities that appear in the same sentence
   are linked with a generic ``RELATED_TO`` relation whose weight is the number
   of co-occurrences. Simple, robust, and language-agnostic.

2. **Dependency-pattern relations** — uses spaCy's dependency parse to find
   subject–verb–object triples, producing typed relations where the verb lemma
   labels the edge (e.g. "model" --uses--> "dataset").

Both return ``Relation`` triples that the knowledge-graph builder consumes
directly.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import List, Optional, Sequence

from src.extraction.entity_extraction import Entity


@dataclass(frozen=True)
class Relation:
    """
    A directed relationship between two entities.

    Attributes
    ----------
    source : str
        Surface form of the head entity.
    relation : str
        Relation type / label (e.g. "RELATED_TO", "uses").
    target : str
        Surface form of the tail entity.
    weight : int
        Strength of the relation (e.g. co-occurrence count).
    """

    source: str
    relation: str
    target: str
    weight: int = 1


@lru_cache(maxsize=2)
def _load_spacy(model_name: str):
    """Load and cache a spaCy model with parser + NER for relation extraction."""
    import spacy

    try:
        return spacy.load(model_name)
    except OSError:  # pragma: no cover - depends on local install
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp


class CooccurrenceRelationExtractor:
    """
    Build ``RELATED_TO`` relations from entities co-occurring in sentences.

    Parameters
    ----------
    spacy_model : str
        spaCy model used only for sentence segmentation.
    window : int
        Number of consecutive sentences treated as one context window.
    relation_label : str
        Label assigned to every co-occurrence edge.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        window: int = 1,
        relation_label: str = "RELATED_TO",
    ) -> None:
        self.window = max(1, window)
        self.relation_label = relation_label
        self._nlp = _load_spacy(spacy_model)

    def extract(self, text: str, entities: Sequence[Entity]) -> List[Relation]:
        """
        Extract co-occurrence relations from one document.

        Parameters
        ----------
        text : str
            Document text (used to locate sentence boundaries).
        entities : sequence of Entity
            Entities previously extracted from the same document.

        Returns
        -------
        list of Relation
            Aggregated, de-duplicated relations with co-occurrence weights.
        """
        if not entities or not text.strip():
            return []

        doc = self._nlp(text)
        sentences = list(doc.sents)
        entity_surfaces = {e.text.lower(): e.text for e in entities}

        pair_counts: Counter = Counter()

        # Slide a window over the sentences and record co-occurring entities
        for start in range(len(sentences)):
            window_text = " ".join(
                s.text for s in sentences[start : start + self.window]
            ).lower()

            present = sorted(
                {original for low, original in entity_surfaces.items() if low in window_text}
            )
            for a, b in combinations(present, 2):
                pair_counts[(a, b)] += 1

        return [
            Relation(source=a, relation=self.relation_label, target=b, weight=count)
            for (a, b), count in pair_counts.items()
        ]


class DependencyRelationExtractor:
    """
    Extract subject–verb–object triples using spaCy dependency parsing.

    The verb lemma becomes the relation label, yielding typed edges such as
    "network" --train--> "model".

    Parameters
    ----------
    spacy_model : str
        spaCy model providing the dependency parse.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        self._nlp = _load_spacy(spacy_model)

    def extract(self, text: str) -> List[Relation]:
        """
        Extract SVO triples from one document.

        Parameters
        ----------
        text : str
            Input document text.

        Returns
        -------
        list of Relation
            One relation per subject–verb–object triple found.
        """
        if not text or not text.strip():
            return []

        doc = self._nlp(text)
        relations: List[Relation] = []

        for sent in doc.sents:
            # Locate the root verb of the sentence
            for token in sent:
                if token.pos_ != "VERB":
                    continue

                subjects = [w for w in token.lefts if w.dep_ in {"nsubj", "nsubjpass"}]
                objects = [w for w in token.rights if w.dep_ in {"dobj", "pobj", "attr", "dative"}]

                for subj in subjects:
                    for obj in objects:
                        subj_phrase = self._expand_span(subj)
                        obj_phrase = self._expand_span(obj)
                        if subj_phrase and obj_phrase:
                            relations.append(
                                Relation(
                                    source=subj_phrase,
                                    relation=token.lemma_.lower(),
                                    target=obj_phrase,
                                    weight=1,
                                )
                            )

        return relations

    @staticmethod
    def _expand_span(token) -> str:
        """Expand a token to its full noun-chunk text where possible."""
        # Include compound modifiers (e.g. "neural network" rather than "network")
        words = [child.text for child in token.lefts if child.dep_ == "compound"]
        words.append(token.text)
        return " ".join(words).strip()


def aggregate_relations(relations: Sequence[Relation]) -> List[Relation]:
    """
    Merge duplicate relations across a corpus, summing their weights.

    Parameters
    ----------
    relations : sequence of Relation
        Relations gathered from many documents.

    Returns
    -------
    list of Relation
        De-duplicated relations sorted by descending weight.
    """
    counter: Counter = Counter()
    for rel in relations:
        counter[(rel.source, rel.relation, rel.target)] += rel.weight

    merged = [
        Relation(source=s, relation=r, target=t, weight=w)
        for (s, r, t), w in counter.items()
    ]
    return sorted(merged, key=lambda rel: rel.weight, reverse=True)
