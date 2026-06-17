from __future__ import annotations

from pathlib import Path


def render_graph_filter(st, graph_path: str = "data/graphs/knowledge_graph.graphml") -> None:
    """Advanced graph filter UI: filter by node degree, label substring, and node type.

    Renders filtered subgraph using pyvis embedding.
    """
    try:
        import networkx as nx
        from pyvis.network import Network
        import tempfile
        import streamlit.components.v1 as components
    except Exception:
        st.error("Optional packages (networkx, pyvis) are required for graph filter view.")
        return

    path = Path(graph_path)
    if not path.exists():
        st.info("Knowledge graph not found. Run pipeline to build the graph.")
        return

    try:
        G = nx.read_graphml(str(path))
    except Exception:
        try:
            import json

            G = nx.node_link_graph(json.loads(path.read_text(encoding="utf-8")))  # type: ignore
        except Exception as exc:
            st.error(f"Failed to load graph: {exc}")
            return

    st.write(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    min_deg = st.slider("Minimum node degree", min_value=0, max_value=50, value=1)
    label_substr = st.text_input("Filter label contains (substring)", value="")
    node_type = st.text_input("Node type (e.g., Paper, Author)", value="")

    nodes = list(G.nodes(data=True))

    def node_matches(n):
        name, attrs = n
        deg = G.degree(name)
        if deg < min_deg:
            return False
        if label_substr and label_substr.lower() not in str(name).lower() and label_substr.lower() not in str(attrs.get("label", "")).lower():
            return False
        if node_type and node_type.lower() not in str(attrs.get("type", "")).lower():
            return False
        return True

    filtered_nodes = [n for n in G.nodes() if node_matches((n, G.nodes[n]))]

    if not filtered_nodes:
        st.info("No nodes matched the filter criteria.")
        return

    # Sampling and paging controls for large graphs
    max_display = st.slider("Max nodes to display (sampling)", min_value=50, max_value=2000, value=500, step=50)
    page_size = st.number_input("Nodes per page", min_value=10, max_value=1000, value=200, step=10)
    total_pages = max(1, (len(filtered_nodes) + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)

    # Determine nodes to show for this page (and respect max_display)
    start = (page - 1) * page_size
    end = start + page_size
    nodes_for_page = filtered_nodes[start:end]
    if len(nodes_for_page) > max_display:
        nodes_for_page = nodes_for_page[:max_display]

    H = G.subgraph(nodes_for_page).copy()

    net = Network(height="600px", width="100%")
    for n, attrs in H.nodes(data=True):
        title = attrs.get("title") or attrs.get("label") or str(n)
        net.add_node(n, label=str(n), title=str(title))

    for u, v, data in H.edges(data=True):
        if u in H and v in H:
            net.add_edge(u, v)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        tmp_path = tmp.name

    html = Path(tmp_path).read_text(encoding="utf-8")
    components.html(html, height=700, scrolling=True)

    # Export / persist controls
    st.markdown("---")
    col_a, col_b = st.columns(2)

    import time

    timestamp = int(time.time())

    with col_a:
        if st.button("Save filtered subgraph as GraphML"):
            out_path = Path("data/graphs") / f"filtered_subgraph_{timestamp}.graphml"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                nx.write_graphml(H, out_path)
                st.success(f"Saved filtered subgraph to {out_path}")
            except Exception as exc:
                st.error(f"Failed to save subgraph: {exc}")

    with col_b:
        if st.button("Export filtered node list (CSV)"):
            out_path = Path("data/graphs") / f"filtered_nodes_{timestamp}.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import csv

                with out_path.open("w", encoding="utf-8", newline="") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(["node_id", "label", "type", "degree"])
                    for n in H.nodes():
                        attrs = H.nodes[n]
                        writer.writerow([n, attrs.get("label", ""), attrs.get("type", ""), H.degree(n)])

                st.success(f"Exported node list to {out_path}")
            except Exception as exc:
                st.error(f"Failed to export node list: {exc}")
