"""Create local silver-standard retrieval queries without title/abstract leakage."""

from pathlib import Path

from src.pipeline.run_pipeline import create_local_relevance_queries


if __name__ == "__main__":
    output = create_local_relevance_queries(
        processed_documents_path=Path("data/processed/processed_documents.csv"),
        output_path=Path("data/evaluation/local_retrieval_queries.json"),
        num_queries=10,
    )
    print(f"Created: {output}")
