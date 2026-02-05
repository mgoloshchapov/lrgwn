import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from lrgwn.layer import CachedSpectralGPSLayer


class SpectralGPSModel(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        K_cheb,
        k_eig,
        heads,
        num_scales: int = 1,
        num_gaussians: int = 16,
        shared_filters: bool = False,
        admissible: bool = False,
        aggregation: str = "concat",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_emb = nn.Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList(
            [
                CachedSpectralGPSLayer(
                    hidden_channels,
                    K_cheb,
                    k_eig,
                    heads=heads,
                    dropout=dropout,
                    num_scales=num_scales,
                    num_gaussians=num_gaussians,
                    shared_filters=shared_filters,
                    admissible=admissible,
                    aggregation=aggregation,
                )
                for _ in range(num_layers)
            ]
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, data):
        x = data.x
        if x is None:
            raise ValueError("Input data is missing node features 'x'.")

        edge_index = data.edge_index
        U = data.U
        Lambda = data.Lambda
        lambda_max = getattr(data, "lambda_max", None)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        x = self.node_emb(x.float())

        for layer in self.layers:
            x = layer(x, edge_index, U, Lambda, lambda_max, batch)

        x = global_mean_pool(x, batch)
        return self.mlp_head(x)
