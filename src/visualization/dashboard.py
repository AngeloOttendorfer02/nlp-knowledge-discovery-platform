from __future__ import annotations

import json
from pathlib import Path


def _load_json(path: Path):
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv(path: Path):
    import pandas as pd

    if not path.exists():
        return None

    try:
        return pd.read_csv(path, dtype={"doc_id": "string"})
    except Exception:
        return None


def _metric_table():
    import pandas as pd

    tables_dir = Path("reports/tables")

    files = {
        "BM25": tables_dir / "bm25_metrics.csv",
        "Semantic": tables_dir / "semantic_metrics.csv",
        "KG-enhanced": tables_dir / "kg_enhanced_metrics.csv",
    }

    frames = []

    for label, path in files.items():
        frame = _read_csv(path)

        if frame is not None and not frame.empty:
            frame = frame.copy()
            frame["method_label"] = label
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _show_metric_chart(st, metrics_df):
    if metrics_df.empty:
        st.info("No retrieval metrics available yet.")
        return

    st.subheader("Retrieval Benchmark Metrics")
    st.dataframe(metrics_df, use_container_width=True)

    metric_options = [
        column
        for column in ["precision_at_k", "recall_at_k", "mrr"]
        if column in metrics_df.columns
    ]

    if not metric_options:
        st.warning("Metric columns were not found in the benchmark tables.")
        return

    selected_metric = st.selectbox(
        "Metric",
        metric_options,
        index=0,
    )

    chart_df = metrics_df.copy()

    if "k" in chart_df.columns:
        chart_df["method_display"] = (
            chart_df["method_label"].astype(str)
            + "@"
            + chart_df["k"].astype(str)
        )
    else:
        chart_df["method_display"] = chart_df["method_label"].astype(str)

    st.bar_chart(
        chart_df.set_index("method_display")[selected_metric]
    )


def _show_results_table(st, title: str, path: Path):
    st.subheader(title)

    frame = _read_csv(path)

    if frame is None or frame.empty:
        st.info(f"No result file found at {path}")
        return

    columns = [
        column
        for column in [
            "method",
            "query_id",
            "query",
            "rank",
            "doc_id",
            "score",
            "base_score",
            "graph_score",
            "relevant",
            "is_relevant",
            "text",
        ]
        if column in frame.columns
    ]

    st.dataframe(
        frame[columns].head(100),
        use_container_width=True,
    )


def run_dashboard(
    processed_path: str = "data/processed/processed_documents.csv",
    graph_summary_path: str = "data/processed/graph_summary.json",
    lda_topics_path: str = "data/processed/lda_topics.json",
    bm25_results_path: str = "data/processed/bm25_search_results.json",
) -> None:
    try:
        import streamlit as st
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Streamlit is required to run the dashboard. "
            "Install it with: pip install streamlit"
        ) from exc

    import pandas as pd

    st.set_page_config(
        page_title="NLP Knowledge Discovery Dashboard",
        layout="wide",
    )

    st.title("NLP Knowledge Discovery Platform")
    st.caption(
        "Scientific document exploration with NLP, retrieval, embeddings, "
        "topic modeling, and knowledge graphs."
    )

    processed = Path(processed_path)
    graph_summary = Path(graph_summary_path)
    lda_topics = Path(lda_topics_path)
    bm25_quick_results = Path(bm25_results_path)

    tabs = st.tabs(
        [
            "Overview",
            "Retrieval Evaluation",
            "Knowledge Graph",
            "Topic Modeling",
            "Search",
            "Files",
        ]
    )

    with tabs[0]:
        st.header("Project Overview")

        df = _read_csv(processed)

        if df is None:
            st.warning(
                "No processed documents found. Run:\n\n"
                "`python -m src.pipeline.run_pipeline --skip-embeddings --full`"
            )
        else:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Documents", len(df))

            with col2:
                st.metric("Columns", len(df.columns))

            with col3:
                if "categories" in df.columns:
                    st.metric("Rows with categories", df["categories"].notna().sum())
                else:
                    st.metric("Rows with categories", 0)

            st.subheader("Processed Documents")
            st.dataframe(df.head(50), use_container_width=True)

            if {"title", "abstract"}.issubset(df.columns):
                stats = pd.DataFrame(
                    [
                        {
                            "metric": "average_title_words",
                            "value": df["title"]
                            .fillna("")
                            .astype(str)
                            .str.split()
                            .map(len)
                            .mean(),
                        },
                        {
                            "metric": "average_abstract_words",
                            "value": df["abstract"]
                            .fillna("")
                            .astype(str)
                            .str.split()
                            .map(len)
                            .mean(),
                        },
                    ]
                )

                st.subheader("Document Statistics")
                st.dataframe(stats, use_container_width=True)

    with tabs[1]:
        st.header("Retrieval Evaluation")

        metrics_df = _metric_table()
        _show_metric_chart(st, metrics_df)

        st.markdown("---")

        tables_dir = Path("reports/tables")

        result_choice = st.selectbox(
            "Result table",
            [
                "BM25",
                "Semantic",
                "KG-enhanced",
            ],
        )

        if result_choice == "BM25":
            _show_results_table(
                st,
                "BM25 Results",
                tables_dir / "bm25_results.csv",
            )
        elif result_choice == "Semantic":
            _show_results_table(
                st,
                "Semantic Retrieval Results",
                tables_dir / "semantic_results.csv",
            )
        else:
            _show_results_table(
                st,
                "KG-enhanced Retrieval Results",
                tables_dir / "kg_enhanced_results.csv",
            )

    with tabs[2]:
        st.header("Knowledge Graph")

        summary = _load_json(graph_summary)

        if summary is None:
            st.info("No graph summary available.")
        else:
            st.json(summary)

        graph_html = Path("reports/figures/knowledge_graph.html")

        if graph_html.exists():
            st.subheader("Interactive Knowledge Graph")
            st.components.v1.html(
                graph_html.read_text(encoding="utf-8"),
                height=700,
                scrolling=True,
            )
        else:
            st.info("No interactive graph visualization available.")

    with tabs[3]:
        st.header("Topic Modeling")

        topics = _load_json(lda_topics)

        if topics is None:
            st.info("No LDA topic output available.")
        else:
            for topic_name, words in topics.items():
                with st.expander(topic_name):
                    st.json(words)

    with tabs[4]:
        st.header("BM25 Quick Search")

        query = st.text_input(
            "Search query",
            value="knowledge graphs and language models",
        )

        top_k = st.slider(
            "Top K",
            min_value=1,
            max_value=20,
            value=10,
        )

        if st.button("Run BM25 Search"):
            df = _read_csv(processed)

            if df is None:
                st.error("Processed documents are missing.")
            else:
                try:
                    from src.retrieval.bm25_retriever import BM25Retriever

                    doc_ids = df["doc_id"].astype(str).tolist()
                    texts = df["text"].fillna("").astype(str).tolist()

                    retriever = BM25Retriever()
                    retriever.index(doc_ids, texts)

                    results = retriever.search(query, top_k=top_k)

                    result_rows = [
                        {
                            "rank": result.rank,
                            "doc_id": result.doc_id,
                            "score": result.score,
                            "text": result.text[:700],
                        }
                        for result in results
                    ]

                    st.dataframe(
                        pd.DataFrame(result_rows),
                        use_container_width=True,
                    )

                except Exception as exc:  # pragma: no cover
                    st.error(f"Search failed: {exc}")

        st.markdown("---")
        st.subheader("Saved BM25 Pipeline Query Output")

        saved = _load_json(bm25_quick_results)

        if saved is None:
            st.info("No saved BM25 quick result available.")
        else:
            st.json(saved[:20])

    with tabs[5]:
        st.header("Generated Files")

        files = [
            Path("data/processed/processed_documents.csv"),
            Path("data/processed/entities.csv"),
            Path("data/processed/keywords.csv"),
            Path("data/processed/relations.csv"),
            Path("data/processed/graph_summary.json"),
            Path("data/processed/lda_topics.json"),
            Path("data/graphs/knowledge_graph.graphml"),
            Path("data/graphs/knowledge_graph.json"),
            Path("reports/figures/knowledge_graph.html"),
            Path("reports/tables/bm25_results.csv"),
            Path("reports/tables/bm25_metrics.csv"),
            Path("reports/tables/semantic_results.csv"),
            Path("reports/tables/semantic_metrics.csv"),
            Path("reports/tables/kg_enhanced_results.csv"),
            Path("reports/tables/kg_enhanced_metrics.csv"),
        ]

        file_rows = [
            {
                "file": str(path),
                "exists": path.exists(),
                "size_kb": round(path.stat().st_size / 1024, 2)
                if path.exists()
                else None,
            }
            for path in files
        ]

        st.dataframe(
            pd.DataFrame(file_rows),
            use_container_width=True,
        )


if __name__ == "__main__":  # pragma: no cover
    run_dashboard()