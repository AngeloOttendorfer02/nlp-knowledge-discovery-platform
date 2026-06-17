from __future__ import annotations

from pathlib import Path
from typing import Dict


def render_topic_viz(st, lda_topics_path: str = "data/processed/lda_topics.json") -> None:
    """Render an interactive topic visualization using Altair.

    The function performs lazy imports so importing this module doesn't require
    optional plotting libraries during tests.
    """
    try:
        import json
        import pandas as pd
        import altair as alt
    except Exception as exc:  # pragma: no cover - interactive
        raise RuntimeError("Altair and pandas are required for topic visualization") from exc

    path = Path(lda_topics_path)

    if not path.exists():
        st.info("LDA topics file not found. Run the pipeline to generate topics.")
        return

    try:
        topics = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - IO
        st.error(f"Failed to load LDA topics: {exc}")
        return

    topic_keys = sorted(topics.keys())
    if not topic_keys:
        st.info("No topics found in topics file.")
        return

    selected = st.selectbox("Select topic to inspect", topic_keys)

    words = topics.get(selected, [])

    if not words:
        st.warning("Selected topic has no words to display.")
        return

    df = pd.DataFrame(words)
    if "word" not in df.columns or "weight" not in df.columns:
        # fallback: attempt to coerce common formats
        if isinstance(words[0], (list, tuple)):
            df = pd.DataFrame(words, columns=["word", "weight"])  # type: ignore
        else:
            st.error("Unexpected topic format in topics file.")
            return

    df = df.sort_values("weight", ascending=False)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("weight:Q", title="Weight"),
            y=alt.Y("word:N", sort=alt.EncodingSortField(field="weight", order="descending")),
            tooltip=["word", "weight"],
        )
        .properties(height=300, width=600)
    )

    st.altair_chart(chart, use_container_width=True)
