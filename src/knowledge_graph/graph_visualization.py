"""
graph_visualization.py — Visualise and analyse the knowledge graph.

Two rendering paths are provided:

1. **Interactive** — an HTML file (via pyvis) that can be opened in a browser
   and explored by dragging nodes; node colour encodes the node type.
2. **Static** — a PNG (via matplotlib + a spring layout) suitable for slides
   and the final report.

A small set of network-analysis helpers (degree distribution, density,
component count) is also included to support the interpretation section.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

# Consistent colour palette per node type (used by both renderers)
_NODE_COLORS: Dict[str, str] = {
    "PAPER": "#4C72B0",
    "AUTHOR": "#DD8452",
    "CONCEPT": "#55A868",
    "TOPIC": "#C44E52",
    "UNKNOWN": "#999999",
}


def _color_for(node_type: str) -> str:
    """Return the palette colour for a node type, falling back to grey."""
    return _NODE_COLORS.get(node_type, _NODE_COLORS["UNKNOWN"])


def _limit_graph(graph, max_nodes: int):
    """
    Return a subgraph with at most ``max_nodes`` highest-degree nodes.

    Visualising very large graphs is unreadable and slow, so we keep the most
    connected nodes, which carry most of the structure.
    """
    if graph.number_of_nodes() <= max_nodes:
        return graph

    degrees = sorted(graph.degree, key=lambda kv: kv[1], reverse=True)
    keep = [node for node, _ in degrees[:max_nodes]]
    return graph.subgraph(keep).copy()


def visualize_interactive(
    graph,
    output_path: str,
    max_nodes: int = 150,
    height: str = "750px",
) -> str:
    """
    Render an interactive HTML visualisation of the knowledge graph.

    Parameters
    ----------
    graph : networkx graph
        The knowledge graph to render.
    output_path : str
        Destination ``.html`` file path.
    max_nodes : int
        Cap on the number of nodes rendered (highest-degree kept).
    height : str
        Canvas height passed to pyvis.

    Returns
    -------
    str
        The path to the written HTML file.
    """
    from pyvis.network import Network

    subgraph = _limit_graph(graph, max_nodes)
    net = Network(height=height, width="100%", directed=True, notebook=False)

    # Add nodes coloured and titled by their type
    for node, data in subgraph.nodes(data=True):
        node_type = data.get("node_type", "UNKNOWN")
        label = data.get("label", str(node))
        net.add_node(
            node,
            label=label,
            color=_color_for(node_type),
            title=f"{node_type}: {label}",
        )

    # Add edges labelled by their relation
    for source, target, data in subgraph.edges(data=True):
        net.add_edge(
            source,
            target,
            title=data.get("relation", ""),
            value=data.get("weight", 1),
        )

    net.force_atlas_2based()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    net.write_html(output_path)
    return output_path


def visualize_static(
    graph,
    output_path: str,
    max_nodes: int = 80,
    figsize: tuple = (14, 10),
    seed: int = 42,
) -> str:
    """
    Render a static PNG visualisation using a spring layout.

    Parameters
    ----------
    graph : networkx graph
        The knowledge graph to render.
    output_path : str
        Destination ``.png`` file path.
    max_nodes : int
        Cap on the number of nodes rendered.
    figsize : tuple
        Matplotlib figure size.
    seed : int
        Random seed for the spring layout (reproducible positions).

    Returns
    -------
    str
        The path to the written image.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    subgraph = _limit_graph(graph, max_nodes)

    # Group node colours by type for a single scatter pass
    node_colors = [
        _color_for(data.get("node_type", "UNKNOWN"))
        for _, data in subgraph.nodes(data=True)
    ]

    pos = nx.spring_layout(subgraph, seed=seed, k=0.5)

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=300, ax=ax)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.3, arrows=True, ax=ax)
    nx.draw_networkx_labels(
        subgraph,
        pos,
        labels={n: d.get("label", n) for n, d in subgraph.nodes(data=True)},
        font_size=7,
        ax=ax,
    )

    # Build a legend from the colour palette
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   markersize=10, label=node_type)
        for node_type, color in _NODE_COLORS.items()
        if node_type != "UNKNOWN"
    ]
    ax.legend(handles=handles, loc="upper right", title="Node type")

    ax.set_title("Knowledge Graph")
    ax.axis("off")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def network_analysis(graph) -> Dict[str, float]:
    """
    Compute summary network statistics for the interpretation section.

    Parameters
    ----------
    graph : networkx graph
        The knowledge graph to analyse.

    Returns
    -------
    dict
        Number of nodes/edges, density, average degree, and component count.
    """
    import networkx as nx

    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    avg_degree = (2 * num_edges / num_nodes) if num_nodes else 0.0

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": nx.density(graph),
        "avg_degree": avg_degree,
        "num_weakly_connected_components": nx.number_weakly_connected_components(graph),
    }
