from __future__ import annotations

import torch
from torch_geometric.transforms import AddLaplacianEigenvectorPE, BaseTransform


def _infer_num_nodes(data) -> int:
    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is not None:
        return int(num_nodes)
    x = getattr(data, "x", None)
    if x is not None:
        return int(x.size(0))
    edge_index = getattr(data, "edge_index", None)
    if edge_index is not None and edge_index.numel() > 0:
        return int(edge_index.max().item()) + 1
    return 0


class SafeAddLaplacianEigenvectorPE(BaseTransform):
    """Adds Laplacian PE and pads/truncates it to a fixed dimension."""

    def __init__(
        self,
        k: int,
        attr_name: str = "laplacian_eigenvector_pe",
        is_undirected: bool = False,
    ) -> None:
        self.k = int(k)
        self.attr_name = attr_name
        self.is_undirected = is_undirected

    def _zeros(self, data, num_nodes: int) -> torch.Tensor:
        edge_index = getattr(data, "edge_index", None)
        device = edge_index.device if edge_index is not None else None
        return torch.zeros((num_nodes, self.k), dtype=torch.float32, device=device)

    def forward(self, data):
        num_nodes = _infer_num_nodes(data)
        if num_nodes <= 0 or self.k <= 0:
            data[self.attr_name] = self._zeros(data, max(num_nodes, 0))
            return data

        k_eff = min(self.k, max(num_nodes - 1, 0))
        if k_eff == 0 or getattr(data, "edge_index", None) is None:
            data[self.attr_name] = self._zeros(data, num_nodes)
            return data

        transform = AddLaplacianEigenvectorPE(
            k=k_eff,
            attr_name=self.attr_name,
            is_undirected=self.is_undirected,
        )
        data = transform(data)
        pe = data[self.attr_name]

        if pe.size(-1) < self.k:
            pe = torch.cat(
                [pe, pe.new_zeros((pe.size(0), self.k - pe.size(-1)))],
                dim=-1,
            )
        elif pe.size(-1) > self.k:
            pe = pe[:, : self.k]

        data[self.attr_name] = pe
        return data
