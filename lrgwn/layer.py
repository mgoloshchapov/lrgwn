import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch_geometric.nn.models.schnet import GaussianSmearing
from torch_geometric.utils import to_dense_batch


class WaveletFilter(nn.Module):
    def __init__(
        self,
        channels: int,
        K_cheb: int,
        num_gaussians: int,
        lambda_cut: float,
        admissible: bool,
        window_type: str | None = "tukey",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.K_cheb = K_cheb
        self.admissible = admissible
        self.lambda_cut = float(lambda_cut)
        self.window_type = window_type

        self.cheb = ChebConv(channels, channels, K=K_cheb, normalization="sym", bias=False)
        self.gaussian_smearing = GaussianSmearing(
            start=0.0,
            stop=lambda_cut,
            num_gaussians=num_gaussians,
        )
        self.spectral_weights = nn.Linear(num_gaussians, channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.cheb.reset_parameters()
        with torch.no_grad():
            for idx, lin in enumerate(self.cheb.lins):
                if idx == 0 and lin.weight.shape[0] == lin.weight.shape[1]:
                    nn.init.eye_(lin.weight)
                elif idx == 0:
                    nn.init.xavier_uniform_(lin.weight)
                else:
                    nn.init.zeros_(lin.weight)
        nn.init.xavier_uniform_(self.spectral_weights.weight)

    def _spatial_zero_response(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for idx, lin in enumerate(self.cheb.lins):
            sign = -1.0 if (idx % 2 == 1) else 1.0
            out = out + sign * lin(x)
        return out

    def _spectral_window(self, values: torch.Tensor) -> torch.Tensor:
        if self.window_type is None:
            return torch.ones_like(values)

        window_key = self.window_type.lower()
        scale = max(self.lambda_cut, 1e-8)
        x = values / scale

        if window_key == "tukey":
            alpha = 0.5
            tail = torch.cos(torch.pi * (x - alpha) / max(2.0 - 2.0 * alpha, 1e-8))
            return torch.where(x <= alpha, torch.ones_like(values), tail.clamp_min(0.0))
        if window_key in {"exp", "exponential"}:
            tau = 1.0 / torch.log(torch.tensor(10.0, device=values.device))
            return torch.exp(-torch.abs(x) / tau)

        raise ValueError(f"Unsupported spectral window '{self.window_type}'.")

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

        # Sanity checks
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
        s_lambda = s_lambda * self._spectral_window(Lambda_scaled).unsqueeze(-1)

        x_dense, mask = to_dense_batch(x, batch)
        U_dense, _ = to_dense_batch(U, batch, max_num_nodes=x_dense.size(1))

        spectral_h = torch.matmul(U_dense.transpose(1, 2), x_dense)
        filtered_h = spectral_h * s_lambda
        out_dense = torch.matmul(U_dense, filtered_h)
        return out_dense[mask]

    def _gaussian_smearing(self, values: torch.Tensor) -> torch.Tensor:
        original_shape = values.shape
        smeared = self.gaussian_smearing(values.reshape(-1))
        return smeared.view(*original_shape, -1)

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
            spatial = spatial - self._spatial_zero_response(x)

        spectral = self._spectral_component(x, U, Lambda, batch, scale)
        return spatial + spectral


class LRGWNLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        K_cheb: int,
        num_scales: int,
        num_gaussians: int,
        lambda_cut: float,
        shared_filters: bool,
        admissible: bool,
        aggregation: str,
        dropout: float,
        residual: bool = True,
        window_type: str | None = "tukey",
        feature_mlp_layers: int = 1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_scales = max(int(num_scales), 1)
        self.shared_filters = shared_filters
        self.aggregation = aggregation
        self.residual = residual

        if feature_mlp_layers <= 0:
            raise ValueError("feature_mlp_layers must be a positive integer.")
        if feature_mlp_layers == 1:
            self.feature_proj = nn.Linear(channels, channels)
        else:
            layers: list[nn.Module] = []
            for _ in range(feature_mlp_layers - 1):
                layers.append(nn.Linear(channels, channels))
                layers.append(nn.GELU())
            layers.append(nn.Linear(channels, channels))
            self.feature_proj = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        if shared_filters:
            self.scale_filter = WaveletFilter(
                channels,
                K_cheb,
                num_gaussians,
                lambda_cut=lambda_cut,
                admissible=False,
                window_type=window_type,
            )
            self.shared_wavelet_filter = WaveletFilter(
                channels,
                K_cheb,
                num_gaussians,
                lambda_cut=lambda_cut,
                admissible=admissible,
                window_type=window_type,
            )
            self.log_s_min = nn.Parameter(torch.tensor(0.0))
            self.log_s_delta = nn.Parameter(torch.tensor(0.0))
        else:
            self.scale_filter = WaveletFilter(
                channels,
                K_cheb,
                num_gaussians,
                lambda_cut=lambda_cut,
                admissible=False,
                window_type=window_type,
            )
            self.wavelet_filters = nn.ModuleList(
                [
                    WaveletFilter(
                        channels,
                        K_cheb,
                        num_gaussians,
                        lambda_cut=lambda_cut,
                        admissible=admissible,
                        window_type=window_type,
                    )
                    for _ in range(self.num_scales)
                ]
            )

        if aggregation not in {"concat", "sum", "mean"}:
            raise ValueError("aggregation must be 'concat', 'sum', or 'mean'")

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
            scale_out = self.scale_filter(
                h, edge_index, U, Lambda, lambda_max, batch, scale=1.0
            )
            outputs.append(self.activation(scale_out))
            for scale in self._scales(x.device):
                wave_out = self.shared_wavelet_filter(
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
        elif self.aggregation == "mean":
            merged = torch.stack(outputs, dim=0).mean(dim=0)
        else:
            merged = self.merge(torch.cat(outputs, dim=-1))

        merged = self.dropout(merged)
        if self.residual:
            merged = x + merged
        return merged
