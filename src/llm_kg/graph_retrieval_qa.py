"""Question-answering helpers that expose graph neighbourhood evidence."""

from __future__ import annotations


def graph_context(graph, node_id: str, depth: int = 1, max_edges: int = 20) -> str:
    """Return a compact textual neighbourhood description for a graph node."""
    if node_id not in graph:
        return ""
    lines: list[str] = []
    frontier = {node_id}
    seen = {node_id}
    for _ in range(max(depth, 0)):
        next_frontier = set()
        for node in frontier:
            for neighbour in graph.neighbors(node):
                if len(lines) >= max_edges:
                    break
                edge_data = graph.get_edge_data(node, neighbour, default={})
                relation = "RELATED_TO"
                if isinstance(edge_data, dict) and edge_data:
                    first = next(iter(edge_data.values())) if all(isinstance(v, dict) for v in edge_data.values()) else edge_data
                    relation = first.get("relation", relation) if isinstance(first, dict) else relation
                lines.append(f"{node} -[{relation}]-> {neighbour}")
                if neighbour not in seen:
                    next_frontier.add(neighbour)
                    seen.add(neighbour)
        frontier = next_frontier
    return "\n".join(lines)


def build_graph_qa_prompt(question: str, graph_text: str) -> str:
    """Create a prompt for answering from graph facts."""
    return f"Use the following knowledge-graph facts to answer.\n\nFacts:\n{graph_text}\n\nQuestion: {question}\nAnswer:"
