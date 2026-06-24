from __future__ import annotations

import pytest

from src.extraction.entity_extraction import Entity, EntityExtractor
from src.extraction.keyword_extraction import TextRankKeywordExtractor, TfidfKeywordExtractor
from src.extraction.relation_extraction import (
    CooccurrenceRelationExtractor,
    Relation,
    aggregate_relations,
)


class _FakeEntity:
    def __init__(self, text: str, label: str, start: int, end: int) -> None:
        self.text = text
        self.label_ = label
        self.start_char = start
        self.end_char = end


class _FakeDoc:
    def __init__(self, entities):
        self.ents = entities


class _FakeNLP:
    def __call__(self, text: str):
        return _FakeDoc(
            [
                _FakeEntity("Graph Neural Networks", "METHOD", 0, 22),
                _FakeEntity("Scientific Retrieval", "TASK", 27, 47),
                _FakeEntity("Graph Neural Networks", "METHOD", 52, 74),
            ]
        )

    def pipe(self, texts, batch_size=64):
        for text in texts:
            if "transformer" in text.lower():
                yield _FakeDoc([_FakeEntity("Transformers", "METHOD", 0, 12)])
            else:
                yield self(text)


def test_entity_extractor_filters_and_deduplicates_entities():
    extractor = EntityExtractor(entity_types=["METHOD"])
    extractor._nlp = _FakeNLP()

    entities = extractor.extract("Graph Neural Networks improve Scientific Retrieval.")

    assert entities == [Entity("Graph Neural Networks", "METHOD", 0, 22)]


def test_entity_extractor_batch_returns_one_result_per_document():
    extractor = EntityExtractor()
    extractor._nlp = _FakeNLP()

    results = extractor.extract_batch(
        [
            "Graph Neural Networks improve retrieval.",
            "Transformers improve named entity recognition.",
        ]
    )

    assert len(results) == 2
    assert results[0][0].text == "Graph Neural Networks"
    assert results[1] == [Entity("Transformers", "METHOD", 0, 12)]


def test_entity_extractor_most_common_entities_counts_across_documents():
    common = EntityExtractor.most_common_entities(
        [
            [Entity("Graph", "CONCEPT", 0, 5), Entity("Retrieval", "TASK", 6, 15)],
            [Entity("Graph", "CONCEPT", 0, 5)],
        ],
        top_n=1,
    )

    assert common == [(('Graph', 'CONCEPT'), 2)]


def test_tfidf_keyword_extractor_returns_ranked_keywords_for_seen_and_unseen_text():
    extractor = TfidfKeywordExtractor(top_n=10, ngram_range=(1, 2))
    extractor.fit(
        [
            "graph neural networks retrieve scientific documents",
            "transformer models extract named entities from papers",
        ]
    )

    seen_keywords = extractor.get_keywords(0)
    unseen_keywords = extractor.get_keywords_for_text("graph retrieval with neural models")

    assert len(seen_keywords) <= 10
    assert seen_keywords[0][1] >= seen_keywords[-1][1]
    assert any("graph" in keyword for keyword, _score in seen_keywords)
    assert any(keyword in {"graph", "neural", "models"} for keyword, _score in unseen_keywords)


def test_tfidf_keyword_extractor_requires_fit_before_lookup():
    extractor = TfidfKeywordExtractor()

    with pytest.raises(RuntimeError, match="Call fit"):
        extractor.get_keywords(0)

    with pytest.raises(RuntimeError, match="Call fit"):
        extractor.get_keywords_for_text("graph retrieval")


def test_textrank_keyword_extractor_handles_empty_and_single_candidate_text():
    extractor = TextRankKeywordExtractor(top_n=5)

    assert extractor.get_keywords("") == []
    # A blank spaCy fallback has no POS tagger, so this test only asserts the
    # stable public contract: a list of ranked keyword-score tuples is returned.
    keywords = extractor.get_keywords("Graph retrieval improves discovery.")
    assert isinstance(keywords, list)
    assert all(isinstance(keyword, str) and isinstance(score, float) for keyword, score in keywords)


def test_cooccurrence_relation_extractor_counts_entities_in_sentence_windows():
    text = "Graph Neural Networks improve Scientific Retrieval. Transformers support Named Entity Recognition."
    entities = [
        Entity("Graph Neural Networks", "METHOD", 0, 22),
        Entity("Scientific Retrieval", "TASK", 31, 51),
        Entity("Transformers", "METHOD", 53, 65),
        Entity("Named Entity Recognition", "TASK", 74, 98),
    ]
    extractor = CooccurrenceRelationExtractor(window=1)

    relations = extractor.extract(text, entities)

    assert Relation("Graph Neural Networks", "RELATED_TO", "Scientific Retrieval", 1) in relations
    assert Relation("Named Entity Recognition", "RELATED_TO", "Transformers", 1) in relations
    assert all(relation.weight == 1 for relation in relations)


def test_cooccurrence_relation_extractor_can_span_multiple_sentences():
    text = "Graph Neural Networks improve retrieval. Scientific Retrieval uses ranking."
    entities = [
        Entity("Graph Neural Networks", "METHOD", 0, 22),
        Entity("Scientific Retrieval", "TASK", 43, 63),
    ]
    extractor = CooccurrenceRelationExtractor(window=2)

    relations = extractor.extract(text, entities)

    assert relations == [Relation("Graph Neural Networks", "RELATED_TO", "Scientific Retrieval", 1)]


def test_aggregate_relations_merges_duplicate_edges_and_sorts_by_weight():
    merged = aggregate_relations(
        [
            Relation("Graph", "RELATED_TO", "Retrieval", 1),
            Relation("Graph", "RELATED_TO", "Retrieval", 2),
            Relation("Transformer", "RELATED_TO", "NER", 1),
        ]
    )

    assert merged[0] == Relation("Graph", "RELATED_TO", "Retrieval", 3)
    assert merged[1] == Relation("Transformer", "RELATED_TO", "NER", 1)
