from __future__ import annotations

from pathlib import Path
from typing import Optional


def render_graph_explorer(st, graph_path: str = "data/graphs/knowledge_graph.graphml") -> None:
    """Render an interactive graph explorer using pyvis and networkx.

    This function does lazy imports to avoid requiring pyvis/networkx during tests.
    """
    try:
        import networkx as nx
        from pyvis.network import Network
        import tempfile
        import json
        import streamlit.components.v1 as components
    except Exception as exc:  # pragma: no cover - interactive
        raise RuntimeError("pyvis and networkx are required for graph explorer") from exc

    path = Path(graph_path)

    if not path.exists():
        st.info("Knowledge graph file not found. Run the pipeline to generate the knowledge graph.")
        return

    try:
        G = nx.read_graphml(str(path))
    except Exception:
        # fallback to reading JSON graph if graphml fails
        try:
            G = nx.node_link_graph(json.loads(path.read_text(encoding="utf-8")))  # type: ignore
        except Exception as exc:  # pragma: no cover - IO
            st.error(f"Failed to load graph: {exc}")
            return

    st.write(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    max_nodes = min(500, G.number_of_nodes())

    net = Network(height="600px", width="100%", notebook=False)

    # Limit nodes for interactive rendering
    nodes_to_add = list(G.nodes())[:max_nodes]
    for n in nodes_to_add:
        attrs = G.nodes[n]
        title = attrs.get("title") or attrs.get("label") or str(n)
        net.add_node(n, label=str(n), title=str(title))

    for u, v, data in list(G.edges(data=True))[: max(0, max_nodes * 5)]:
        net.add_edge(u, v)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        tmp_path = tmp.name

    # Read and embed HTML
    html = Path(tmp_path).read_text(encoding="utf-8")
    components.html(html, height=700, scrolling=True)
