import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from layer import CachedSpectralGPSLayer

class SpectralGPSModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, K_cheb, k_eig, heads):
        super().__init__()
        self.node_emb = nn.Linear(in_channels, hidden_channels)
        
        self.layers = nn.ModuleList([
            CachedSpectralGPSLayer(hidden_channels, K_cheb, k_eig, heads)
            for _ in range(num_layers)
        ])
        
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        U, Lambda, l_max = data.U, data.Lambda, data.lambda_max
        
        x = self.node_emb(x.float())
        
        for layer in self.layers:
            x = layer(x, edge_index, U, Lambda, l_max, batch)
        
        # Global Pooling for Graph Classification (Peptides-func)
        x = global_mean_pool(x, batch)
        return self.mlp_head(x)