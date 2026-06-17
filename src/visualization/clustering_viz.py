from __future__ import annotations

from pathlib import Path


def render_clustering(st, embeddings_path: str = "artifacts/semantic_retrieval/document_embeddings.npz") -> None:
    """Render document clustering based on precomputed embeddings (NPZ expected).

    The NPZ should contain either an array named 'embeddings' and optional 'ids',
    or be a single array saved as the first element.
    """
    try:
        import numpy as np
        import pandas as pd
        import umap
        import hdbscan
        import altair as alt
    except Exception:  # pragma: no cover - interactive
        st.error("Optional packages (numpy, umap, hdbscan, altair) are required for clustering view.")
        return

    path = Path(embeddings_path)
    if not path.exists():
        st.info(f"Embeddings file not found at {path}. Run embedding step in pipeline.")
        return

    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:  # pragma: no cover - IO
        st.error(f"Failed to load embeddings: {exc}")
        return

    if "embeddings" in data:
        embeddings = data["embeddings"]
        ids = data["ids"] if "ids" in data else [str(i) for i in range(len(embeddings))]
    else:
        # try to infer
        arrs = [data[k] for k in data.files]
        embeddings = arrs[0]
        ids = arrs[1] if len(arrs) > 1 else [str(i) for i in range(len(embeddings))]

    if embeddings.ndim != 2:
        st.error("Unexpected embeddings shape; expected 2D array")
        return

    reducer = umap.UMAP(n_components=2, random_state=42)
    proj = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    labels = clusterer.fit_predict(proj)

    df = pd.DataFrame({"id": [str(i) for i in ids], "x": proj[:, 0], "y": proj[:, 1], "cluster": labels})

    chart = alt.Chart(df).mark_circle(size=60).encode(
        x="x:Q",
        y="y:Q",
        color=alt.Color("cluster:N", legend=alt.Legend(title="Cluster")),
        tooltip=["id", "cluster"],
    ).properties(width=700, height=500)

    st.altair_chart(chart, use_container_width=True)
    st.write(df.groupby("cluster").size().rename("count"))
