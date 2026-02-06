import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from lrgwn.layer import LRGWNLayer


class SpectralGPSModel(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        K_cheb,
        num_scales: int,
        num_gaussians: int,
        lambda_cut: float,
        shared_filters: bool,
        admissible: bool,
        aggregation: str,
        dropout: float,
        positional_dim: int,
        structural_dim: int,
        positional_attr: str,
        structural_attr: str,
    ):
        super().__init__()
        self.node_emb = nn.Linear(in_channels, hidden_channels)
        self.positional_attr = positional_attr
        self.structural_attr = structural_attr
        self.positional_encoder = (
            nn.Linear(positional_dim, hidden_channels) if positional_dim > 0 else None
        )
        self.structural_encoder = (
            nn.Linear(structural_dim, hidden_channels) if structural_dim > 0 else None
        )

        self.wavelet_layers = nn.ModuleList(
            [
                LRGWNLayer(
                    channels=hidden_channels,
                    K_cheb=K_cheb,
                    dropout=dropout,
                    num_scales=num_scales,
                    num_gaussians=num_gaussians,
                    lambda_cut=lambda_cut,
                    shared_filters=shared_filters,
                    admissible=admissible,
                    aggregation=aggregation,
                )
                for _ in range(num_layers)
            ]
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        if x is None:
            x = torch.ones((data.num_nodes, self.node_emb.in_features), device=edge_index.device)
        elif x.dim() == 1:
            x = x.unsqueeze(-1)

        U = data.U
        Lambda = data.Lambda
        lambda_max = getattr(data, "lambda_max", None)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        x = self.node_emb(x.float())

        if self.positional_encoder is not None:
            if not hasattr(data, self.positional_attr):
                raise ValueError(
                    f"Missing positional encoding attribute '{self.positional_attr}' in data."
                )
            x = x + self.positional_encoder(getattr(data, self.positional_attr).float())

        if self.structural_encoder is not None:
            if not hasattr(data, self.structural_attr):
                raise ValueError(
                    f"Missing structural encoding attribute '{self.structural_attr}' in data."
                )
            x = x + self.structural_encoder(getattr(data, self.structural_attr).float())

        for layer in self.wavelet_layers:
            x = layer(x, edge_index, U, Lambda, lambda_max, batch)

        x = global_mean_pool(x, batch)
        return self.mlp_head(x)
