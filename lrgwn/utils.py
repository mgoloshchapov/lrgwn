import torch
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from scipy.sparse.linalg import eigsh
from torch_geometric.transforms import BaseTransform


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
