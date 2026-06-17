from __future__ import annotations

from pathlib import Path


def render_topic_embeddings(st, lda_topics_path: str = "data/processed/lda_topics.json") -> None:
    """Render 2D embedding projection of topic words using SentenceTransformers + UMAP.

    Lazy imports ensure tests won't require these optional libraries.
    """
    try:
        import json
        import pandas as pd
        from sentence_transformers import SentenceTransformer
        import umap
        import altair as alt
    except Exception as exc:  # pragma: no cover - interactive
        st.error("Optional packages (sentence-transformers, umap, altair) are required for topic embeddings view.")
        return

    path = Path(lda_topics_path)
    if not path.exists():
        st.info("LDA topics not found. Run pipeline to generate topics.")
        return

    try:
        topics = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - IO
        st.error(f"Failed to read topics: {exc}")
        return

    # Flatten words across topics
    records = []
    for tname, words in topics.items():
        for w in words:
            if isinstance(w, dict):
                word = w.get("word")
                weight = w.get("weight")
            elif isinstance(w, (list, tuple)) and len(w) >= 2:
                word, weight = w[0], w[1]
            else:
                word, weight = str(w), 1.0

            records.append({"topic": tname, "word": str(word), "weight": float(weight)})

    df = pd.DataFrame.from_records(records)
    df = df.drop_duplicates(subset=["word"]).reset_index(drop=True)

    model_name = st.selectbox("SentenceTransformer model", ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"], index=0)
    model = SentenceTransformer(model_name)

    embeddings = model.encode(df["word"].tolist(), show_progress_bar=False)

    # Nearest-neighbors support
    try:
        from sklearn.neighbors import NearestNeighbors
        import numpy as np
    except Exception:  # pragma: no cover - interactive
        st.warning("sklearn required for nearest-neighbors; nearest neighbor search will be disabled.")
        NearestNeighbors = None  # type: ignore

    reducer = umap.UMAP(n_components=2, random_state=42)
    proj = reducer.fit_transform(embeddings)

    df_proj = df.copy()
    df_proj["x"] = proj[:, 0]
    df_proj["y"] = proj[:, 1]

    chart = alt.Chart(df_proj).mark_circle(size=60).encode(
        x="x:Q",
        y="y:Q",
        color="topic:N",
        tooltip=["word", "topic", "weight"],
    ).properties(width=700, height=450)

    st.altair_chart(chart, use_container_width=True)

    # Interactive label search and nearest neighbors
    query = st.text_input("Search for a word (exact) or pick from list", value="")
    word_list = df["word"].tolist()
    picked = None
    if query:
        if query in word_list:
            picked = query
        else:
            # try case-insensitive match
            matches = [w for w in word_list if query.lower() in w.lower()]
            if matches:
                picked = st.selectbox("Matches", matches)

    if picked:
        st.markdown(f"**Selected word:** {picked}")
        idx = df_proj.index[df_proj["word"] == picked].tolist()
        if idx:
            i = idx[0]
            st.write(df_proj.loc[i:i])

            if NearestNeighbors is not None:
                nn = NearestNeighbors(n_neighbors=6, metric="cosine").fit(embeddings)
                distances, indices = nn.kneighbors([embeddings[i]])
                neighbors = [word_list[k] for k in indices[0] if k != i]
                st.write({"neighbors": neighbors, "distances": distances[0].tolist()})
