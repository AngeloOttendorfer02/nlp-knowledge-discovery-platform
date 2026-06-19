"""Training helpers for optional GNN experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GCNTrainingResult:
    """Container returned by train_gcn."""

    model: object
    losses: list[float]
    final_loss: float | None
    predictions: np.ndarray
    accuracy: float


def normalized_adjacency(num_nodes: int, edge_index: np.ndarray) -> np.ndarray:
    """Return symmetric normalized adjacency with self-loops."""
    adjacency = np.eye(num_nodes, dtype="float32")

    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.size:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, num_edges).")
        for src, dst in edge_index.T:
            adjacency[int(src), int(dst)] = 1.0
            adjacency[int(dst), int(src)] = 1.0

    degree = adjacency.sum(axis=1)
    degree[degree == 0] = 1.0

    inv_sqrt_degree = np.diag(1.0 / np.sqrt(degree))
    return inv_sqrt_degree @ adjacency @ inv_sqrt_degree


def train_gcn(
    features,
    edge_index,
    labels,
    epochs: int = 10,
    hidden_dim: int = 16,
    learning_rate: float = 0.01,
    seed: int = 42,
) -> GCNTrainingResult:
    """Train a small SimpleGCN model on node features and labels."""
    try:
        import torch
        from torch import nn, optim
    except Exception as exc:
        raise ImportError(
            "train_gcn requires PyTorch. Install torch to run GNN experiments."
        ) from exc

    from src.gnn.gcn_model import SimpleGCN

    torch.manual_seed(seed)
    np.random.seed(seed)

    features_np = np.asarray(features, dtype=np.float32)
    edge_index_np = np.asarray(edge_index, dtype=np.int64)
    labels_np = np.asarray(labels, dtype=np.int64)

    if features_np.ndim != 2:
        raise ValueError("features must be a 2D array.")
    if edge_index_np.ndim != 2 or edge_index_np.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges).")
    if labels_np.ndim != 1 or labels_np.shape[0] != features_np.shape[0]:
        raise ValueError("labels must be a 1D array with one label per node.")
    if epochs < 0:
        raise ValueError("epochs must be non-negative.")

    adjacency_np = normalized_adjacency(
        num_nodes=features_np.shape[0],
        edge_index=edge_index_np,
    )

    x = torch.tensor(features_np, dtype=torch.float32)
    adjacency = torch.tensor(adjacency_np, dtype=torch.float32)
    y = torch.tensor(labels_np, dtype=torch.long)

    model = SimpleGCN(
        input_dim=features_np.shape[1],
        hidden_dim=hidden_dim,
        output_dim=int(labels_np.max()) + 1,
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    losses: list[float] = []

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x, adjacency)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        logits = model(x, adjacency)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    accuracy = float(np.mean(predictions == labels_np))

    return GCNTrainingResult(
        model=model,
        losses=losses,
        final_loss=losses[-1] if losses else None,
        predictions=predictions,
        accuracy=accuracy,
    )
