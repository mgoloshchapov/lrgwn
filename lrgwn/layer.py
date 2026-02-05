import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch_geometric.utils import to_dense_batch


class WaveletFilter(nn.Module):
    def __init__(
        self,
        channels: int,
        K_cheb: int,
        num_gaussians: int,
        lambda_cut: float = 2.0,
        admissible: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.K_cheb = K_cheb
        self.admissible = admissible

        self.cheb = ChebConv(channels, channels, K=K_cheb, normalization="sym")
        offsets = torch.linspace(0.0, lambda_cut, num_gaussians)
        delta = lambda_cut / max(num_gaussians - 1, 1)
        coeff = -0.5 / (delta * delta)
        self.register_buffer("gaussian_offset", offsets)
        self.register_buffer("gaussian_coeff", torch.tensor(coeff))
        self.spectral_weights = nn.Linear(num_gaussians, channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.cheb.reset_parameters()
        for idx, lin in enumerate(self.cheb.lins):
            if lin.weight.shape[0] == lin.weight.shape[1]:
                if idx == 0:
                    nn.init.eye_(lin.weight)
                else:
                    nn.init.zeros_(lin.weight)
            else:
                nn.init.zeros_(lin.weight)
        if self.cheb.bias is not None:
            nn.init.zeros_(self.cheb.bias)
        nn.init.ones_(self.spectral_weights.weight)

    def _spectral_component(
        self,
        x: torch.Tensor,
        U: torch.Tensor,
        Lambda: torch.Tensor,
        batch: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        if U.numel() == 0:
            return torch.zeros_like(x)

        if Lambda.dim() == 1:
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            if Lambda.numel() % num_graphs == 0:
                Lambda = Lambda.view(num_graphs, -1)
            else:
                Lambda = Lambda.unsqueeze(0)
        elif Lambda.dim() != 2:
            raise ValueError(f"Lambda must be 1D or 2D, got shape {Lambda.shape}.")

        Lambda_scaled = Lambda * scale
        s_lambda = self._gaussian_smearing(Lambda_scaled)
        if self.admissible:
            s0 = self._gaussian_smearing(
                torch.zeros(1, device=Lambda.device, dtype=Lambda.dtype)
            )
            s_lambda = s_lambda - s0

        s_lambda = self.spectral_weights(s_lambda)

        x_dense, mask = to_dense_batch(x, batch)
        U_dense, _ = to_dense_batch(U, batch, max_num_nodes=x_dense.size(1))

        spectral_h = torch.matmul(U_dense.transpose(1, 2), x_dense)
        filtered_h = spectral_h * s_lambda
        out_dense = torch.matmul(U_dense, filtered_h)
        return out_dense[mask]

    def _gaussian_smearing(self, values: torch.Tensor) -> torch.Tensor:
        diff = values.unsqueeze(-1) - self.gaussian_offset
        return torch.exp(self.gaussian_coeff * diff * diff)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        U: torch.Tensor,
        Lambda: torch.Tensor,
        lambda_max: torch.Tensor,
        batch: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        lambda_max_scaled = None
        if lambda_max is not None:
            lambda_max_scaled = lambda_max / scale

        spatial = self.cheb(
            x,
            edge_index,
            lambda_max=lambda_max_scaled,
            batch=batch,
        )
        if self.admissible:
            spatial = spatial - self.cheb.lins[0](x)

        spectral = self._spectral_component(x, U, Lambda, batch, scale)
        return spatial + spectral


class LRGWNLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        K_cheb: int,
        k_eig: int,
        num_scales: int = 1,
        num_gaussians: int = 16,
        shared_filters: bool = False,
        admissible: bool = False,
        aggregation: str = "concat",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_scales = max(int(num_scales), 1)
        self.shared_filters = shared_filters
        self.aggregation = aggregation

        self.feature_proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        if shared_filters:
            self.shared_filter = WaveletFilter(
                channels,
                K_cheb,
                num_gaussians,
                admissible=admissible,
            )
            self.log_s_min = nn.Parameter(torch.tensor(0.0))
            self.log_s_delta = nn.Parameter(torch.tensor(0.0))
        else:
            self.scale_filter = WaveletFilter(
                channels,
                K_cheb,
                num_gaussians,
                admissible=admissible,
            )
            self.wavelet_filters = nn.ModuleList(
                [
                    WaveletFilter(
                        channels,
                        K_cheb,
                        num_gaussians,
                        admissible=admissible,
                    )
                    for _ in range(self.num_scales)
                ]
            )

        if aggregation not in {"concat", "sum"}:
            raise ValueError("aggregation must be 'concat' or 'sum'")

        if aggregation == "concat":
            self.merge = nn.Linear((self.num_scales + 1) * channels, channels)
        else:
            self.merge = None

    def _scales(self, device: torch.device) -> torch.Tensor:
        s_min = torch.exp(self.log_s_min)
        s_max = s_min + F.softplus(self.log_s_delta)
        if self.num_scales == 1:
            return s_max.unsqueeze(0)
        log_s_min = torch.log(s_min)
        log_s_max = torch.log(s_max)
        steps = torch.linspace(0.0, 1.0, self.num_scales, device=device)
        log_s = log_s_max + (log_s_min - log_s_max) * steps
        return torch.exp(log_s)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        U: torch.Tensor,
        Lambda: torch.Tensor,
        lambda_max: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = self.feature_proj(x)
        outputs = []

        if self.shared_filters:
            scale_out = self.shared_filter(
                h, edge_index, U, Lambda, lambda_max, batch, scale=1.0
            )
            outputs.append(self.activation(scale_out))
            for scale in self._scales(x.device):
                wave_out = self.shared_filter(
                    h, edge_index, U, Lambda, lambda_max, batch, scale=scale
                )
                outputs.append(self.activation(wave_out))
        else:
            scale_out = self.scale_filter(
                h, edge_index, U, Lambda, lambda_max, batch, scale=1.0
            )
            outputs.append(self.activation(scale_out))
            for filt in self.wavelet_filters:
                wave_out = filt(h, edge_index, U, Lambda, lambda_max, batch, scale=1.0)
                outputs.append(self.activation(wave_out))

        if self.aggregation == "sum":
            merged = torch.stack(outputs, dim=0).sum(dim=0)
        else:
            merged = self.merge(torch.cat(outputs, dim=-1))

        return self.dropout(merged)


class CachedSpectralGPSLayer(LRGWNLayer):
    def __init__(
        self,
        channels,
        K_cheb,
        k_eig,
        heads=4,
        dropout=0.1,
        num_scales: int = 1,
        num_gaussians: int = 16,
        shared_filters: bool = False,
        admissible: bool = False,
        aggregation: str = "concat",
    ):
        super().__init__(
            channels=channels,
            K_cheb=K_cheb,
            k_eig=k_eig,
            num_scales=num_scales,
            num_gaussians=num_gaussians,
            shared_filters=shared_filters,
            admissible=admissible,
            aggregation=aggregation,
            dropout=dropout,
        )
