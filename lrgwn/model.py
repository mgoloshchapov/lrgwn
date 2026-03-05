import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from lrgwn.layer import LRGWNLayer


def _build_raw_norm(norm_type: str | None, dim: int) -> nn.Module | None:
    if dim <= 0 or norm_type is None:
        return None
    norm_key = norm_type.lower()
    if norm_key == "batchnorm":
        return nn.BatchNorm1d(dim)
    raise ValueError(f"Unsupported raw_norm_type='{norm_type}'.")


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
        encoding_fusion: str = "add",
        positional_out_dim: int | None = None,
        structural_out_dim: int | None = None,
        positional_raw_norm_type: str | None = None,
        structural_raw_norm_type: str | None = None,
    ):
        super().__init__()
        self.encoding_fusion = encoding_fusion
        if self.encoding_fusion not in {"add", "concat"}:
            raise ValueError("encoding_fusion must be either 'add' or 'concat'.")

        pos_out_dim = 0 if positional_dim <= 0 else (positional_out_dim or 0)
        struct_out_dim = 0 if structural_dim <= 0 else (structural_out_dim or 0)

        if self.encoding_fusion == "add":
            self.node_emb = nn.Linear(in_channels, hidden_channels)
            if positional_dim > 0:
                self.positional_encoder = nn.Linear(positional_dim, hidden_channels)
            else:
                self.positional_encoder = None
            if structural_dim > 0:
                self.structural_encoder = nn.Linear(structural_dim, hidden_channels)
            else:
                self.structural_encoder = None
        else:
            if positional_dim > 0 and pos_out_dim <= 0:
                raise ValueError(
                    "model.positional_encoding.output_dim must be > 0 for concat fusion."
                )
            if structural_dim > 0 and struct_out_dim <= 0:
                raise ValueError(
                    "model.structural_encoding.output_dim must be > 0 for concat fusion."
                )
            base_dim = hidden_channels - pos_out_dim - struct_out_dim
            if base_dim <= 0:
                raise ValueError(
                    "hidden_channels must be larger than positional+structural output dims."
                )
            self.node_emb = nn.Linear(in_channels, base_dim)
            self.positional_encoder = (
                nn.Linear(positional_dim, pos_out_dim) if positional_dim > 0 else None
            )
            self.structural_encoder = (
                nn.Linear(structural_dim, struct_out_dim) if structural_dim > 0 else None
            )

        self.positional_raw_norm = _build_raw_norm(positional_raw_norm_type, positional_dim)
        self.structural_raw_norm = _build_raw_norm(structural_raw_norm_type, structural_dim)

        self.positional_attr = positional_attr
        self.structural_attr = structural_attr

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

        if self.encoding_fusion == "add":
            if self.positional_encoder is not None:
                if not hasattr(data, self.positional_attr):
                    raise ValueError(
                        f"Missing positional encoding attribute '{self.positional_attr}' in data."
                    )
                pos = getattr(data, self.positional_attr).float()
                if self.positional_raw_norm is not None:
                    pos = self.positional_raw_norm(pos)
                x = x + self.positional_encoder(pos)

            if self.structural_encoder is not None:
                if not hasattr(data, self.structural_attr):
                    raise ValueError(
                        f"Missing structural encoding attribute '{self.structural_attr}' in data."
                    )
                se = getattr(data, self.structural_attr).float()
                if self.structural_raw_norm is not None:
                    se = self.structural_raw_norm(se)
                x = x + self.structural_encoder(se)
        else:
            parts = [x]
            if self.positional_encoder is not None:
                if not hasattr(data, self.positional_attr):
                    raise ValueError(
                        f"Missing positional encoding attribute '{self.positional_attr}' in data."
                    )
                pos = getattr(data, self.positional_attr).float()
                if self.positional_raw_norm is not None:
                    pos = self.positional_raw_norm(pos)
                parts.append(self.positional_encoder(pos))

            if self.structural_encoder is not None:
                if not hasattr(data, self.structural_attr):
                    raise ValueError(
                        f"Missing structural encoding attribute '{self.structural_attr}' in data."
                    )
                se = getattr(data, self.structural_attr).float()
                if self.structural_raw_norm is not None:
                    se = self.structural_raw_norm(se)
                parts.append(self.structural_encoder(se))
            x = torch.cat(parts, dim=-1)

        for layer in self.wavelet_layers:
            x = layer(x, edge_index, U, Lambda, lambda_max, batch)

        x = global_mean_pool(x, batch)
        return self.mlp_head(x)
