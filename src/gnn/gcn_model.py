"""Optional PyTorch GCN model used for extension experiments."""

from __future__ import annotations


def torch_available() -> bool:
    """Return True when PyTorch can be imported."""
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


if torch_available():
    import torch
    from torch import nn

    class SimpleGCN(nn.Module):
        """Small dependency-light GCN using dense normalized adjacency."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
            h = adjacency @ x
            h = torch.relu(self.linear1(h))
            h = self.dropout(h)
            h = adjacency @ h
            return self.linear2(h)

else:

    class SimpleGCN:  # type: ignore[no-redef]
        """Placeholder that explains the optional dependency clearly."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "SimpleGCN requires PyTorch. Install torch to run GNN experiments."
            )