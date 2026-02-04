import torch
import torch.nn as nn
from torch_geometric.nn import GPSConv, ChebConv

class SpectralLocalNN(nn.Module):
    def __init__(self, channels, K_cheb, k_eig, gamma=1.0):
        super().__init__()
        # Dual MLPs for feature separation
        self.mlp_spatial = nn.Sequential(
            nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, channels)
        )
        self.mlp_spectral = nn.Sequential(
            nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, channels)
        )
        
        self.cheb_local = ChebConv(channels, channels, K=K_cheb)
        self.gamma = nn.Parameter(torch.tensor([gamma]))
        self.spectral_proj = nn.Linear(channels, channels, bias=False)

    def forward(self, x, edge_index, U, Lambda, lambda_max=None, **kwargs):
        # 1. Spatial Path
        x_sp = self.mlp_spatial(x)
        out_local = self.cheb_local(x_sp, edge_index, lambda_max=lambda_max)
        
        # 2. Spectral Path (Gaussian Smearing)
        x_spec = self.mlp_spectral(x)
        h_proj = self.spectral_proj(x_spec)
        s_lambda = torch.exp(-self.gamma * (Lambda ** 2))
        
        # Spectral Projection: U S(Lambda) U^T H
        # U is [N_batch, k], h_proj is [N_batch, channels]
        spectral_h = torch.matmul(U.t(), h_proj)      # [k, channels]
        filtered_h = s_lambda.unsqueeze(1) * spectral_h
        out_spectral = torch.matmul(U, filtered_h)    # [N_batch, channels]
        
        return out_local + out_spectral

class CachedSpectralGPSLayer(nn.Module):
    def __init__(self, channels, K_cheb, k_eig, heads=4, dropout=0.1):
        super().__init__()
        self.local_nn = SpectralLocalNN(channels, K_cheb, k_eig)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.conv = GPSConv(channels, self.local_nn, attn=self.attn, dropout=dropout)

    def forward(self, x, edge_index, U, Lambda, lambda_max, batch):
        # kwargs are passed to SpectralLocalNN
        return self.conv(x, edge_index, batch, U=U, Lambda=Lambda, lambda_max=lambda_max)