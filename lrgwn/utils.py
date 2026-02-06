from __future__ import annotations

import torch
from scipy.sparse.linalg import eigsh
from torch_geometric.transforms import (
    AddLaplacianEigenvectorPE,
    AddRandomWalkPE,
    BaseTransform,
    Compose,
    LaplacianLambdaMax,
)
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix


class CachedSpectralTransform(BaseTransform):
    def __init__(self, k):
        self.k = k

    def forward(self, data):
        num_nodes = data.num_nodes
        if num_nodes is None:
            if getattr(data, "x", None) is not None:
                num_nodes = data.x.size(0)
            elif getattr(data, "edge_index", None) is not None and data.edge_index.numel() > 0:
                num_nodes = int(data.edge_index.max()) + 1
            else:
                num_nodes = 0
        num_nodes = int(num_nodes)

        k = int(self.k)
        k_eff = min(k, max(num_nodes - 1, 0))
        if k_eff == 0:
            data.Lambda = torch.zeros((1, k), dtype=torch.float32)
            data.U = torch.zeros((num_nodes, k), dtype=torch.float32)
            return data

        edge_index = getattr(data, "edge_index", None)
        if edge_index is None or edge_index.numel() == 0:
            lambdas = torch.zeros(k, dtype=torch.float32)
            U = torch.zeros((num_nodes, k), dtype=torch.float32)
            if k_eff > 0:
                U[:, :k_eff] = torch.eye(num_nodes, dtype=torch.float32)[:, :k_eff]
            data.Lambda = lambdas.unsqueeze(0)
            data.U = U
            return data

        # Small epsilon to handle disconnected graphs
        edge_index, edge_weight = get_laplacian(
            edge_index, normalization="sym", num_nodes=num_nodes
        )
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

        try:
            lambdas, U = eigsh(L, k=k_eff, which="LM", sigma=1e-6)
            order = lambdas.argsort()
            lambdas = lambdas[order]
            U = U[:, order]
            lambdas = torch.from_numpy(lambdas).float()
            U = torch.from_numpy(U).float()
        except Exception:
            L_dense = torch.from_numpy(L.toarray()).float()
            lambdas, U = torch.linalg.eigh(L_dense)
            lambdas = lambdas[:k_eff].float()
            U = U[:, :k_eff].float()

        if k_eff < k:
            padded_lambdas = torch.zeros(k, dtype=lambdas.dtype)
            padded_lambdas[:k_eff] = lambdas
            padded_U = torch.zeros((num_nodes, k), dtype=U.dtype)
            padded_U[:, :k_eff] = U
            data.Lambda = padded_lambdas.unsqueeze(0)
            data.U = padded_U
        else:
            data.Lambda = lambdas.unsqueeze(0)
            data.U = U

        return data


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


def _compose_or_none(transforms):
    if not transforms:
        return None
    if len(transforms) == 1:
        return transforms[0]
    return Compose(transforms)


def build_dataset_transforms(cfg):
    base_transforms = [CachedSpectralTransform(k=cfg.dataset.k_eig)]
    if cfg.dataset.use_lambda_max:
        base_transforms.append(LaplacianLambdaMax(normalization="sym"))
    encoding_transforms = []

    pos_dim = 0
    pos_cfg = cfg.model.positional_encoding
    if pos_cfg.enabled:
        if pos_cfg.kind == "laplacian_eigenvector":
            encoding_transforms.append(
                SafeAddLaplacianEigenvectorPE(
                    k=pos_cfg.dim,
                    attr_name=pos_cfg.attr_name,
                    is_undirected=pos_cfg.is_undirected,
                )
            )
        elif pos_cfg.kind == "random_walk":
            encoding_transforms.append(
                AddRandomWalkPE(
                    walk_length=pos_cfg.dim,
                    attr_name=pos_cfg.attr_name,
                )
            )
        else:
            raise ValueError(f"Unsupported positional_encoding.kind='{pos_cfg.kind}'.")
        pos_dim = pos_cfg.dim

    struct_dim = 0
    struct_cfg = cfg.model.structural_encoding
    if struct_cfg.enabled:
        if struct_cfg.kind == "random_walk":
            encoding_transforms.append(
                AddRandomWalkPE(
                    walk_length=struct_cfg.dim,
                    attr_name=struct_cfg.attr_name,
                )
            )
        elif struct_cfg.kind == "laplacian_eigenvector":
            encoding_transforms.append(
                SafeAddLaplacianEigenvectorPE(
                    k=struct_cfg.dim,
                    attr_name=struct_cfg.attr_name,
                    is_undirected=True,
                )
            )
        else:
            raise ValueError(f"Unsupported structural_encoding.kind='{struct_cfg.kind}'.")
        struct_dim = struct_cfg.dim

    if pos_cfg.enabled and struct_cfg.enabled and pos_cfg.attr_name == struct_cfg.attr_name:
        raise ValueError(
            "positional_encoding.attr_name and structural_encoding.attr_name must differ."
        )

    if cfg.dataset.precompute_transforms:
        pre_transform = _compose_or_none(base_transforms + encoding_transforms)
        transform = None
    else:
        pre_transform = None
        transform = _compose_or_none(base_transforms + encoding_transforms)

    return pre_transform, transform, pos_dim, struct_dim
