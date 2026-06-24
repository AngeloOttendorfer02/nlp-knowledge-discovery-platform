"""
End-to-end NLP Knowledge Discovery Pipeline.

Run from the project root:

    python -m src.pipeline.run_pipeline

Fast first run:

    python -m src.pipeline.run_pipeline --skip-embeddings

Run with the real arXiv Kaggle metadata file:

    python -m src.pipeline.run_pipeline --input data/raw/arxiv-metadata-oai-snapshot.json --skip-embeddings

Run the full project workflow including retrieval experiments:

    python -m src.pipeline.run_pipeline --skip-embeddings --full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

from src.extraction.entity_extraction import EntityExtractor
from src.extraction.keyword_extraction import TfidfKeywordExtractor
from src.extraction.relation_extraction import (
    CooccurrenceRelationExtractor,
    aggregate_relations,
)
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from src.knowledge_graph.graph_visualization import (
    network_analysis,
    visualize_interactive,
)
from src.preprocessing.document_loader import (
    dataframe_to_documents,
    load_arxiv_csv,
    load_arxiv_jsonl,
)
from src.preprocessing.text_cleaning import preprocess, preprocess_to_string
from src.retrieval.bm25_retriever import BM25Retriever
from src.topic_modeling.lda_model import LDATopicModel


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def ensure_directories(config: Dict[str, Any]) -> Dict[str, Path]:
    paths = config.get("paths", {})

    directories = {
        "raw": Path(paths.get("data_raw", "data/raw")),
        "processed": Path(paths.get("data_processed", "data/processed")),
        "graphs": Path(paths.get("data_graphs", "data/graphs")),
        "reports": Path(paths.get("reports", "reports")),
        "figures": Path(paths.get("figures", "reports/figures")),
        "tables": Path("reports/tables"),
        "evaluation": Path("data/evaluation"),
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


def create_sample_dataset(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    sample = pd.DataFrame(
        [
            {
                "id": "sample-001",
                "title": "Knowledge Graphs for Scientific Discovery",
                "abstract": (
                    "Knowledge graphs represent entities and relations in scientific literature. "
                    "They support semantic search, question answering, and explainable AI systems."
                ),
                "authors": "Alice Smith, Bob Miller",
                "categories": "cs.AI cs.CL",
                "update_date": "2026-01-01",
            },
            {
                "id": "sample-002",
                "title": "Graph Neural Networks for Link Prediction",
                "abstract": (
                    "Graph neural networks learn representations over graph-structured data. "
                    "They can predict missing links and classify nodes in citation networks."
                ),
                "authors": "Carla Brown",
                "categories": "cs.LG cs.AI",
                "update_date": "2026-01-02",
            },
            {
                "id": "sample-003",
                "title": "Bias Analysis in Language Model Embeddings",
                "abstract": (
                    "Large language models and embedding spaces may contain social bias. "
                    "Bias evaluation and debiasing methods are important for responsible NLP."
                ),
                "authors": "David Wilson",
                "categories": "cs.CL",
                "update_date": "2026-01-03",
            },
            {
                "id": "sample-004",
                "title": "Topic Modeling for Research Trend Analysis",
                "abstract": (
                    "Topic modeling discovers hidden themes in scientific document collections. "
                    "LDA and transformer-based methods can support literature review workflows."
                ),
                "authors": "Eva Johnson",
                "categories": "cs.IR cs.CL",
                "update_date": "2026-01-04",
            },
            {
                "id": "sample-005",
                "title": "Retrieval Augmented Generation with Structured Knowledge",
                "abstract": (
                    "Retrieval augmented generation combines external knowledge retrieval with "
                    "language generation. Knowledge graphs can provide structured evidence."
                ),
                "authors": "Frank Davis",
                "categories": "cs.AI cs.IR",
                "update_date": "2026-01-05",
            },
        ]
    )

    sample.to_csv(path, index=False)
    return path


def find_input_file(raw_dir: Path) -> Optional[Path]:
    for pattern in ["*.csv", "*.jsonl", "*.json"]:
        matches = sorted(raw_dir.glob(pattern))
        if matches:
            return matches[0]

    return None


def load_documents(input_path: Optional[str], config: Dict[str, Any], raw_dir: Path):
    dataset_cfg = config.get("dataset", {})
    sample_size = dataset_cfg.get("sample_size")
    categories = dataset_cfg.get("categories")
    seed = config.get("project", {}).get("seed", 42)

    if input_path:
        path = Path(input_path)
    else:
        path = find_input_file(raw_dir)

        if path is None:
            path = create_sample_dataset(raw_dir / "sample_documents.csv")
            print(f"No raw dataset found. Created sample dataset at: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = load_arxiv_csv(
            str(path),
            sample_size=sample_size,
            categories=categories,
            seed=seed,
        )
    elif suffix in {".jsonl", ".json"}:
        df = load_arxiv_jsonl(
            str(path),
            sample_size=sample_size,
            categories=categories,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")

    documents = dataframe_to_documents(df)
    return documents, path


def documents_to_dataframe(documents) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "abstract": doc.abstract,
                "authors": doc.authors,
                "categories": doc.categories,
                "date": doc.date,
                "text": doc.text,
            }
            for doc in documents
        ]
    )


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _simple_query_terms(text: str, max_terms: int = 4) -> list[str]:
    """Return compact content terms without copying title or abstract spans."""
    import re

    stopwords = {
        "about", "after", "also", "analysis", "and", "are", "based", "been",
        "between", "can", "data", "document", "documents", "for", "from", "has",
        "have", "into", "method", "methods", "model", "models", "paper", "papers",
        "present", "propose", "research", "results", "show", "such", "that", "the",
        "their", "these", "this", "using", "with", "within", "which", "will",
    }
    tokens = [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z-]{3,}", text)]
    counts: dict[str, int] = {}
    for token in tokens:
        token = token.strip("-")
        if token in stopwords or len(token) < 4:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts, key=lambda term: (-counts[term], term))
    return ranked[:max_terms]


def create_local_relevance_queries(
    processed_documents_path: Path,
    output_path: Path,
    num_queries: int = 10,
) -> Path:
    """Create reproducible local retrieval queries without title/abstract leakage.

    The previous smoke-test query file copied each target paper's title plus the
    beginning of its abstract. That is useful for a pipeline smoke test, but it
    can inflate retrieval metrics. This generator instead creates compact topic
    style queries from categories and high-frequency content terms. The relevant
    document ID is still silver-standard and local, but the query no longer
    contains the exact title or abstract passage of the target paper.
    """
    if not processed_documents_path.exists():
        raise FileNotFoundError(f"Processed documents not found: {processed_documents_path}")

    df = pd.read_csv(processed_documents_path, dtype={"doc_id": "string"})
    required_columns = {"doc_id", "text"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"processed_documents.csv is missing required columns: {missing}")

    df = df.copy()
    df["doc_id"] = df["doc_id"].fillna("").astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    if "categories" not in df.columns:
        df["categories"] = ""
    df["categories"] = df["categories"].fillna("").astype(str).str.strip()

    candidates = df[(df["doc_id"] != "") & (df["text"].str.len() >= 80)].copy()
    if candidates.empty:
        raise ValueError("Could not create local relevance queries because no suitable documents were found.")

    candidates = candidates.head(num_queries)
    queries = []

    for index, (_, row) in enumerate(candidates.iterrows(), start=1):
        terms = _simple_query_terms(row["text"], max_terms=4)
        import re

        raw_categories = str(row["categories"])
        category_terms = []
        for category in re.findall(r"[A-Za-z]{2}(?:\.[A-Za-z]{2})?", raw_categories):
            category_terms.extend(category.replace(".", " ").split())
        category_terms = category_terms[:2]

        query_parts = ["scientific papers about", *terms]
        if category_terms:
            query_parts.extend(["in", *category_terms])
        query_text = " ".join(query_parts).strip()

        queries.append(
            {
                "query_id": f"q{index}",
                "query": query_text,
                "relevant_doc_ids": [row["doc_id"]],
                "notes": "Local silver query generated without copying the target title or abstract passage.",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(queries, file, indent=2, ensure_ascii=False)

    print(f"Saved local relevance queries to {output_path}")
    return output_path
    

def run_retrieval_experiments(
    documents_path: Path,
    queries_path: Path,
    graph_path: Path,
    output_dir: Path,
) -> None:
    """Run all retrieval benchmarks after the core pipeline."""
    if not documents_path.exists():
        raise FileNotFoundError(f"Missing processed documents: {documents_path}")

    if not queries_path.exists():
        raise FileNotFoundError(f"Missing evaluation queries: {queries_path}")

    if not graph_path.exists():
        raise FileNotFoundError(f"Missing knowledge graph: {graph_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    retrieval_cfg = config.get("retrieval", {})
    embeddings_cfg = config.get("embeddings", {})
    kg_cfg = config.get("knowledge_graph", {})

    top_k = int(retrieval_cfg.get("top_k", 10))
    top_ks = sorted({5, top_k})

    from src.experiments.run_bm25_baseline import run_bm25_baseline
    from src.experiments.run_semantic_retrieval import run_semantic_experiment
    from src.experiments.run_kg_enhanced_retrieval import run_kg_enhanced_experiment

    print("Running BM25 baseline experiment...")

    run_bm25_baseline(
        documents_path=documents_path,
        queries_path=queries_path,
        output_dir=output_dir,
        bm25_k1=float(retrieval_cfg.get("bm25_k1", 1.5)),
        bm25_b=float(retrieval_cfg.get("bm25_b", 0.75)),
        top_ks=top_ks,
    )

    print("Running semantic retrieval experiment...")

    run_semantic_experiment(
        documents_path=documents_path,
        queries_path=queries_path,
        output_dir=output_dir,
        model_name=str(
            retrieval_cfg.get(
                "embedding_model",
                embeddings_cfg.get("model", "all-MiniLM-L6-v2"),
            )
        ),
        batch_size=int(embeddings_cfg.get("batch_size", 32)),
        top_ks=top_ks,
        cache_path="artifacts/semantic_retrieval/document_embeddings.npz",
        force_recompute=False,
    )

    print("Running knowledge graph-enhanced retrieval experiment...")

    run_kg_enhanced_experiment(
        documents_path=documents_path,
        queries_path=queries_path,
        graph_path=graph_path,
        output_dir=output_dir,
        base_method="semantic",
        model_name=str(
            retrieval_cfg.get(
                "embedding_model",
                embeddings_cfg.get("model", "all-MiniLM-L6-v2"),
            )
        ),
        batch_size=int(embeddings_cfg.get("batch_size", 32)),
        bm25_k1=float(retrieval_cfg.get("bm25_k1", 1.5)),
        bm25_b=float(retrieval_cfg.get("bm25_b", 0.75)),
        top_ks=top_ks,
        retrieval_weight=float(kg_cfg.get("retrieval_weight", 0.75)),
        graph_weight=float(kg_cfg.get("graph_weight", 0.25)),
        candidate_multiplier=int(kg_cfg.get("candidate_multiplier", 3)),
        expansion_weight=float(kg_cfg.get("expansion_weight", 0.5)),
        keywords_path="data/processed/keywords.csv",
        keyword_top_n=int(config.get("extraction", {}).get("keyword_top_n", 10)),
        cache_path="artifacts/semantic_retrieval/document_embeddings.npz",
        force_recompute=False,
    )

    print("All retrieval experiments completed successfully.")


def run_pipeline(
    input_path: Optional[str] = None,
    query: str = "knowledge graphs and language models",
    skip_embeddings: bool = False,
) -> Dict[str, Path]:
    print("Starting NLP Knowledge Discovery Pipeline...")

    config = load_config()
    directories = ensure_directories(config)

    print("Loading documents...")
    documents, used_input_path = load_documents(input_path, config, directories["raw"])
    print(f"Loaded {len(documents)} documents from {used_input_path}")

    if not documents:
        raise RuntimeError("No documents loaded. Add a CSV, JSON, or JSONL file to data/raw.")

    df = documents_to_dataframe(documents)

    texts = df["text"].fillna("").tolist()
    doc_ids = df["doc_id"].astype(str).tolist()

    print("Preprocessing text...")
    preprocessing_cfg = config.get("preprocessing", {})

    tokenized_docs = [
        preprocess(
            text,
            spacy_model=preprocessing_cfg.get("spacy_model", "en_core_web_sm"),
            remove_stopwords=preprocessing_cfg.get("remove_stopwords", True),
            lemmatize=preprocessing_cfg.get("lemmatize", True),
            min_token_length=preprocessing_cfg.get("min_token_length", 3),
        )
        for text in texts
    ]

    cleaned_texts = [
        preprocess_to_string(
            text,
            spacy_model=preprocessing_cfg.get("spacy_model", "en_core_web_sm"),
            remove_stopwords=preprocessing_cfg.get("remove_stopwords", True),
            lemmatize=preprocessing_cfg.get("lemmatize", True),
            min_token_length=preprocessing_cfg.get("min_token_length", 3),
        )
        for text in texts
    ]

    df["cleaned_text"] = cleaned_texts
    df["tokens"] = [" ".join(tokens) for tokens in tokenized_docs]

    processed_path = directories["processed"] / "processed_documents.csv"
    df.to_csv(processed_path, index=False)
    print(f"Saved processed documents to {processed_path}")

    print("Extracting entities...")
    extraction_cfg = config.get("extraction", {})

    entity_extractor = EntityExtractor(
        spacy_model=extraction_cfg.get("spacy_model", "en_core_web_sm"),
        entity_types=extraction_cfg.get("entity_types"),
    )

    doc_entities = entity_extractor.extract_batch(texts)

    entities_output = []

    for doc_id, entities in zip(doc_ids, doc_entities):
        for entity in entities:
            entities_output.append(
                {
                    "doc_id": doc_id,
                    "text": entity.text,
                    "label": entity.label,
                    "start": entity.start,
                    "end": entity.end,
                }
            )

    entities_path = directories["processed"] / "entities.csv"
    pd.DataFrame(entities_output).to_csv(entities_path, index=False)
    print(f"Saved extracted entities to {entities_path}")

    print("Extracting keywords...")

    keyword_extractor = TfidfKeywordExtractor(
        top_n=extraction_cfg.get("keyword_top_n", 10)
    )
    keyword_extractor.fit(cleaned_texts)

    keywords_output = []

    for index, doc_id in enumerate(doc_ids):
        for keyword, score in keyword_extractor.get_keywords(index):
            keywords_output.append(
                {
                    "doc_id": doc_id,
                    "keyword": keyword,
                    "score": score,
                }
            )

    keywords_path = directories["processed"] / "keywords.csv"
    pd.DataFrame(keywords_output).to_csv(keywords_path, index=False)
    print(f"Saved extracted keywords to {keywords_path}")

    print("Extracting co-occurrence relations...")

    relation_extractor = CooccurrenceRelationExtractor(
        spacy_model=extraction_cfg.get("spacy_model", "en_core_web_sm"),
        window=extraction_cfg.get("relation_window", 1),
    )

    all_relations = []
    relations_output = []

    for doc_id, text, entities in zip(doc_ids, texts, doc_entities):
        relations = relation_extractor.extract(text, entities)
        all_relations.extend(relations)

        for relation in relations:
            relations_output.append(
                {
                    "doc_id": doc_id,
                    "source": relation.source,
                    "relation": relation.relation,
                    "target": relation.target,
                    "weight": relation.weight,
                }
            )

    aggregated_relations = aggregate_relations(all_relations)

    relations_path = directories["processed"] / "relations.csv"
    pd.DataFrame(relations_output).to_csv(relations_path, index=False)
    print(f"Saved extracted relations to {relations_path}")

    print("Building knowledge graph...")
    kg_cfg = config.get("knowledge_graph", {})

    graph_builder = KnowledgeGraphBuilder(
        min_edge_weight=kg_cfg.get("min_edge_weight", 1),
    )

    for doc, entities in zip(documents, doc_entities):
        graph_builder.add_paper(
            doc_id=doc.doc_id,
            title=doc.title,
            authors=doc.authors,
            categories=doc.categories,
        )
        graph_builder.add_entities(doc.doc_id, entities)

    graph_builder.add_relations(aggregated_relations)
    graph_builder.prune()

    graphml_path = directories["graphs"] / "knowledge_graph.graphml"
    graph_json_path = directories["graphs"] / "knowledge_graph.json"

    graph_builder.save_graphml(str(graphml_path))
    graph_builder.save_json(str(graph_json_path))

    graph_stats = graph_builder.stats()
    graph_analysis = network_analysis(graph_builder.graph)

    save_json(
        {
            "graph_stats": graph_stats,
            "network_analysis": graph_analysis,
        },
        directories["processed"] / "graph_summary.json",
    )

    print(f"Saved knowledge graph to {graphml_path}")
    print(f"Knowledge graph stats: {graph_stats}")

    print("Creating knowledge graph visualization...")
    graph_html_path = directories["figures"] / "knowledge_graph.html"

    visualize_interactive(
        graph_builder.graph,
        output_path=str(graph_html_path),
        max_nodes=kg_cfg.get("max_nodes_visualize", 150),
    )

    print(f"Saved graph visualization to {graph_html_path}")

    print("Building BM25 retrieval index...")
    retrieval_cfg = config.get("retrieval", {})

    bm25 = BM25Retriever(
        k1=retrieval_cfg.get("bm25_k1", 1.5),
        b=retrieval_cfg.get("bm25_b", 0.75),
    )

    bm25.index(doc_ids, texts)
    bm25_results = bm25.search(query, top_k=retrieval_cfg.get("top_k", 10))

    save_json(
        [
            {
                "rank": result.rank,
                "doc_id": result.doc_id,
                "score": result.score,
                "text": result.text,
            }
            for result in bm25_results
        ],
        directories["processed"] / "bm25_search_results.json",
    )

    print("Saved BM25 search results.")

    print("Running LDA topic modeling...")
    topic_cfg = config.get("topic_modeling", {})
    num_topics = min(topic_cfg.get("num_topics", 10), max(1, len(documents)))

    try:
        lda = LDATopicModel(
            num_topics=num_topics,
            passes=topic_cfg.get("lda_passes", 10),
            iterations=topic_cfg.get("lda_iterations", 50),
            seed=config.get("project", {}).get("seed", 42),
        )

        lda.fit(tokenized_docs)
        topics = lda.get_topics(top_n_words=10)

        save_json(
            {
                f"topic_{idx}": [
                    {"word": word, "weight": weight}
                    for word, weight in topic_words
                ]
                for idx, topic_words in enumerate(topics)
            },
            directories["processed"] / "lda_topics.json",
        )

        print("Saved LDA topics.")

    except Exception as exc:
        print(f"Skipping LDA topic modeling because it failed: {exc}")

    if not skip_embeddings:
        print("Building embeddings and semantic network...")

        try:
            import networkx as nx

            from src.embeddings.network_embeddings import SemanticNetworkBuilder
            from src.embeddings.sentence_embeddings import SentenceEmbedder

            embedding_cfg = config.get("embeddings", {})

            embedder = SentenceEmbedder(
                model_name=embedding_cfg.get("model", "all-MiniLM-L6-v2"),
                batch_size=embedding_cfg.get("batch_size", 32),
            )

            embedder.fit(doc_ids, texts)

            semantic_builder = SemanticNetworkBuilder(
                similarity_threshold=embedding_cfg.get("similarity_threshold", 0.5),
            )

            semantic_graph = semantic_builder.build_knn_network(
                doc_ids=doc_ids,
                embeddings=embedder.embeddings,
                k=min(3, max(1, len(doc_ids) - 1)),
            )

            semantic_graph_path = directories["graphs"] / "semantic_network.graphml"
            nx.write_graphml(semantic_graph, semantic_graph_path)

            semantic_stats = semantic_builder.network_stats(semantic_graph)

            save_json(
                semantic_stats,
                directories["processed"] / "semantic_network_summary.json",
            )

            print(f"Saved semantic network to {semantic_graph_path}")

        except Exception as exc:
            print(f"Skipping embedding/semantic network step because it failed: {exc}")

    print("Pipeline completed successfully.")

    return {
        "processed_documents": processed_path,
        "entities": entities_path,
        "keywords": keywords_path,
        "relations": relations_path,
        "knowledge_graph_graphml": graphml_path,
        "knowledge_graph_json": graph_json_path,
        "knowledge_graph_html": graph_html_path,
        "reports_tables": directories["tables"],
        "evaluation_dir": directories["evaluation"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the NLP Knowledge Discovery Pipeline."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional input CSV, JSON, or JSONL file. If omitted, data/raw is searched.",
    )

    parser.add_argument(
        "--query",
        type=str,
        default="knowledge graphs and language models",
        help="Example retrieval query used for BM25 output.",
    )

    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding and semantic-network creation for a faster first run.",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full workflow: core pipeline, auto queries, and all retrieval experiments.",
    )

    parser.add_argument(
        "--run-experiments",
        action="store_true",
        help="Run BM25, semantic, and KG-enhanced retrieval experiments after the pipeline.",
    )

    parser.add_argument(
        "--auto-create-queries",
        action="store_true",
        help="Create local evaluation queries from the processed documents.",
    )

    parser.add_argument(
        "--num-auto-queries",
        type=int,
        default=10,
        help="Number of local relevance queries to create automatically.",
    )

    parser.add_argument(
        "--queries",
        type=str,
        default="data/evaluation/local_retrieval_queries.json",
        help="Path to the evaluation query JSON file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    outputs = run_pipeline(
        input_path=args.input,
        query=args.query,
        skip_embeddings=args.skip_embeddings,
    )

    queries_path = Path(args.queries)

    if args.full or args.auto_create_queries:
        create_local_relevance_queries(
            processed_documents_path=outputs["processed_documents"],
            output_path=queries_path,
            num_queries=args.num_auto_queries,
        )

    if args.full or args.run_experiments:
        run_retrieval_experiments(
            documents_path=outputs["processed_documents"],
            queries_path=queries_path,
            graph_path=outputs["knowledge_graph_graphml"],
            output_dir=outputs["reports_tables"],
        )


if __name__ == "__main__":
    main()