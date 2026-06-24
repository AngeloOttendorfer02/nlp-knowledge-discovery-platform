"""
test_regression_fixes.py — Regression tests for recently fixed issues.

These tests lock in the behaviour of four fixes so they cannot silently break
again:

1. Rule-based entity fallback no longer captures ordinary words (case-sensitive
   capitalisation patterns + stopword trimming).
2. The spaCy fallback emits a clear RuntimeWarning when the configured model is
   missing.
3. The FAISS-backed vector store always persists a NumPy copy, so a store can be
   reloaded and searched even without FAISS.
4. GCN training tolerates ``-1`` (unlabeled) nodes by masking them out of the
   loss and the accuracy computation.
"""

import os
import tempfile
import warnings

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fix 1: rule-based entity fallback must not capture ordinary words
# ---------------------------------------------------------------------------

def test_rule_based_fallback_ignores_ordinary_words():
    """Ordinary lowercase words must not be returned as entities."""
    from src.extraction.entity_extraction import _rule_based_entities

    text = "We use models for research and propose methods that show results."
    entities = _rule_based_entities(text)

    surfaces = {e.text.lower() for e in entities}
    ordinary = {"use", "for", "research", "propose", "show", "results", "models", "we"}
    assert not (surfaces & ordinary), f"ordinary words leaked as entities: {surfaces & ordinary}"


def test_rule_based_fallback_keeps_scientific_terms():
    """Genuine scientific terms and acronyms must still be detected."""
    from src.extraction.entity_extraction import _rule_based_entities

    text = (
        "We propose Graph Neural Networks for knowledge graphs. "
        "Our BERT model uses retrieval augmented generation with FAISS."
    )
    surfaces = {e.text.lower() for e in _rule_based_entities(text)}

    for expected in [
        "graph neural networks",
        "knowledge graphs",
        "bert",
        "faiss",
        "retrieval augmented generation",
    ]:
        assert expected in surfaces, f"expected scientific term missing: {expected}"

    # Leading determiners must have been trimmed away
    assert "our" not in surfaces and "we" not in surfaces


def test_rule_based_fallback_offsets_match_after_trimming():
    """start/end must point at the trimmed surface, not the original match span.

    Regression for the case where a leading stopword is removed (e.g.
    "Our BERT" -> "BERT"): the offsets must move past the trimmed prefix so that
    text[start:end] equals the returned entity text.
    """
    from src.extraction.entity_extraction import _rule_based_entities

    texts = [
        "Our BERT model uses retrieval augmented generation with FAISS.",
        "We propose Graph Neural Networks for knowledge graphs.",
        "The Transformer model and BERT outperform LSTM networks.",
    ]
    for text in texts:
        for entity in _rule_based_entities(text):
            assert text[entity.start:entity.end] == entity.text, (
                f"offset mismatch: {entity.text!r} vs text[{entity.start}:{entity.end}]"
                f"={text[entity.start:entity.end]!r}"
            )


# ---------------------------------------------------------------------------
# Fix 2: spaCy fallback warns when the model is missing
# ---------------------------------------------------------------------------

def test_missing_spacy_model_emits_runtime_warning():
    """Constructing an extractor with an unavailable model must warn, not crash."""
    pytest.importorskip("spacy")
    from src.extraction.entity_extraction import EntityExtractor

    # A model name that is guaranteed not to be installed. Using a unique name
    # also avoids the lru_cache returning a previously loaded pipeline.
    fake_model = "nonexistent_model_for_regression_test_xyz"

    with pytest.warns(RuntimeWarning):
        extractor = EntityExtractor(spacy_model=fake_model)

    # The fallback pipeline must still produce entities from scientific text
    entities = extractor.extract("Graph Neural Networks improve knowledge graphs.")
    assert len(entities) > 0


# ---------------------------------------------------------------------------
# Fix 3: vector store persists a NumPy fallback and reloads/searches correctly
# ---------------------------------------------------------------------------

def test_vector_store_persists_numpy_fallback_and_reloads():
    """Saving must write vectors.npy; reloading must allow searching again."""
    from src.retrieval.vector_store import VectorStore

    embeddings = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]], dtype="float32"
    )

    store = VectorStore(dim=3)
    store.add(["a", "b", "c"], embeddings)

    with tempfile.TemporaryDirectory() as tmp:
        store.save(tmp)

        # The NumPy fallback copy must always be written alongside the index
        assert os.path.exists(os.path.join(tmp, "vectors.npy")), "vectors.npy was not persisted"

        # A reloaded store must return the identical-vector document first
        reloaded = VectorStore.load(tmp)
        hits = reloaded.search(np.array([1.0, 0.0, 0.0], dtype="float32"), top_k=2)
        assert hits and hits[0].doc_id == "a"


def test_vector_store_numpy_backend_search():
    """The pure-NumPy backend (use_faiss=False) must rank by cosine similarity."""
    from src.retrieval.vector_store import VectorStore

    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype="float32")
    store = VectorStore(dim=2, use_faiss=False)
    store.add(["x", "y", "z"], embeddings)

    hits = store.search(np.array([1.0, 0.0], dtype="float32"), top_k=3)
    assert hits[0].doc_id == "x"  # identical direction → highest similarity


# ---------------------------------------------------------------------------
# Fix 4: GCN training tolerates -1 (unlabeled) nodes
# ---------------------------------------------------------------------------

def test_train_gcn_handles_unlabeled_nodes():
    """Nodes labelled -1 must be ignored by the loss instead of crashing."""
    pytest.importorskip("torch")
    from src.gnn.train_gnn import train_gcn

    features = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype="float32"
    )
    edge_index = np.array([[0, 1, 2, 3], [1, 0, 3, 2]], dtype="int64")
    # Two labelled nodes (classes 0 and 1) and two unlabeled nodes (-1)
    labels = np.array([0, -1, 1, -1], dtype="int64")

    result = train_gcn(features, edge_index, labels, epochs=3, hidden_dim=4, seed=1)

    assert len(result.losses) == 3
    assert result.predictions.shape == (4,)
    # Loss must be finite (no NaN from feeding -1 into the loss)
    assert all(np.isfinite(loss) for loss in result.losses)


def test_train_gcn_rejects_all_unlabeled():
    """If every node is unlabeled, training must raise a clear error."""
    pytest.importorskip("torch")
    from src.gnn.train_gnn import train_gcn

    features = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    edge_index = np.array([[0, 1], [1, 0]], dtype="int64")
    labels = np.array([-1, -1], dtype="int64")

    with pytest.raises(ValueError):
        train_gcn(features, edge_index, labels, epochs=2, hidden_dim=4, seed=1)
