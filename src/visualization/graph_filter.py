from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def _get_node_type(attributes: Mapping[str, Any]) -> str:
    """Return the canonical graph node type with backward compatibility.

    KnowledgeGraphBuilder stores the type in ``node_type``. The fallback to
    ``type`` keeps the visualization compatible with older exported graphs.
    """

    return str(attributes.get("node_type", attributes.get("type", "")))


def render_graph_filter(
    st,
    graph_path: str = "data/graphs/knowledge_graph.graphml",
) -> None:
    """Render an interactive filtered view of the project knowledge graph.

    Nodes can be filtered by minimum degree, label substring, and canonical
    node type. The resulting subgraph can be displayed, saved as GraphML, or
    exported as a CSV node list.
    """

    try:
        import tempfile
        import time

        import networkx as nx
        import streamlit.components.v1 as components
        from pyvis.network import Network
    except Exception:
        st.error(
            "Optional packages (networkx and pyvis) are required for the "
            "graph filter view."
        )
        return

    path = Path(graph_path)
    if not path.exists():
        st.info("Knowledge graph not found. Run the pipeline to build the graph.")
        return

    try:
        graph = nx.read_graphml(str(path))
    except Exception:
        try:
            import json

            graph = nx.node_link_graph(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except Exception as exc:
            st.error(f"Failed to load graph: {exc}")
            return

    st.write(
        f"Graph: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
    )

    min_degree = st.slider(
        "Minimum node degree",
        min_value=0,
        max_value=50,
        value=1,
    )
    label_substring = st.text_input(
        "Filter label contains (substring)",
        value="",
    )
    node_type_filter = st.text_input(
        "Node type (for example PAPER, AUTHOR, TOPIC, or CONCEPT)",
        value="",
    )

    def node_matches(node_id: str, attributes: Mapping[str, Any]) -> bool:
        if graph.degree(node_id) < min_degree:
            return False

        label = str(attributes.get("label", node_id))
        if label_substring:
            normalized_filter = label_substring.casefold()
            if (
                normalized_filter not in str(node_id).casefold()
                and normalized_filter not in label.casefold()
            ):
                return False

        if node_type_filter:
            if node_type_filter.casefold() not in _get_node_type(
                attributes
            ).casefold():
                return False

        return True

    filtered_nodes = [
        node_id
        for node_id, attributes in graph.nodes(data=True)
        if node_matches(node_id, attributes)
    ]

    if not filtered_nodes:
        st.info("No nodes matched the filter criteria.")
        return

    max_display = st.slider(
        "Max nodes to display (sampling)",
        min_value=50,
        max_value=2000,
        value=500,
        step=50,
    )
    page_size = st.number_input(
        "Nodes per page",
        min_value=10,
        max_value=1000,
        value=200,
        step=10,
    )
    total_pages = max(1, (len(filtered_nodes) + page_size - 1) // page_size)
    page = st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=1,
    )

    start = (page - 1) * page_size
    end = start + page_size
    nodes_for_page = filtered_nodes[start:end][:max_display]
    subgraph = graph.subgraph(nodes_for_page).copy()

    network = Network(height="600px", width="100%")
    for node_id, attributes in subgraph.nodes(data=True):
        label = str(attributes.get("label", node_id))
        title = attributes.get("title") or label
        node_type_value = _get_node_type(attributes)
        tooltip = f"{node_type_value}: {title}" if node_type_value else str(title)
        network.add_node(
            node_id,
            label=label,
            title=tooltip,
        )

    for source, target in subgraph.edges():
        network.add_edge(source, target)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
        network.save_graph(temp_file.name)
        temporary_path = Path(temp_file.name)

    try:
        html = temporary_path.read_text(encoding="utf-8")
        components.html(html, height=700, scrolling=True)
    finally:
        temporary_path.unlink(missing_ok=True)

    st.markdown("---")
    graphml_column, csv_column = st.columns(2)
    timestamp = int(time.time())

    with graphml_column:
        if st.button("Save filtered subgraph as GraphML"):
            output_path = (
                Path("data/graphs")
                / f"filtered_subgraph_{timestamp}.graphml"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                nx.write_graphml(subgraph, output_path)
                st.success(f"Saved filtered subgraph to {output_path}")
            except Exception as exc:
                st.error(f"Failed to save subgraph: {exc}")

    with csv_column:
        if st.button("Export filtered node list (CSV)"):
            output_path = (
                Path("data/graphs")
                / f"filtered_nodes_{timestamp}.csv"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import csv

                with output_path.open(
                    "w",
                    encoding="utf-8",
                    newline="",
                ) as file_handle:
                    writer = csv.writer(file_handle)
                    # Keep the historic CSV header "type" for compatibility,
                    # but export the canonical node_type value.
                    writer.writerow(["node_id", "label", "type", "degree"])
                    for node_id, attributes in subgraph.nodes(data=True):
                        writer.writerow(
                            [
                                node_id,
                                attributes.get("label", ""),
                                _get_node_type(attributes),
                                subgraph.degree(node_id),
                            ]
                        )

                st.success(f"Exported node list to {output_path}")
            except Exception as exc:
                st.error(f"Failed to export node list: {exc}")
