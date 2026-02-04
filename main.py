import hydra
from omegaconf import DictConfig
import torch
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, LaplacianLambdaMax, BaseTransform
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from scipy.sparse.linalg import eigsh

from lrgwn.model import SpectralGPSModel

class CachedSpectralTransform(BaseTransform):
    def __init__(self, k=20):
        self.k = k
    def __call__(self, data):
        # Small epsilon to handle disconnected graphs
        edge_index, edge_weight = get_laplacian(data.edge_index, normalization='sym', num_nodes=data.num_nodes)
        L = to_scipy_sparse_matrix(edge_index, edge_weight, data.num_nodes)
        lambdas, U = eigsh(L, k=self.k, which='LM', sigma=1e-6)
        data.Lambda = torch.from_numpy(lambdas).float()
        data.U = torch.from_numpy(U).float()
        return data

@hydra.main(config_path="configs/peptides-func.yaml", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    
    # 1. Load Dataset with Spectral Transforms
    transform = Compose([CachedSpectralTransform(k=cfg.dataset.k_eig), LaplacianLambdaMax(normalization='sym')])
    train_dataset = LRGBDataset(root=cfg.dataset.root, name=cfg.dataset.name, split='train', pre_transform=transform)
    val_dataset = LRGBDataset(root=cfg.dataset.root, name=cfg.dataset.name, split='val', pre_transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size)

    # 2. Setup Model
    model = SpectralGPSModel(
        in_channels=train_dataset.num_features,
        hidden_channels=cfg.model.hidden_channels,
        out_channels=train_dataset.num_classes,
        num_layers=cfg.model.num_layers,
        K_cheb=cfg.model.K_cheb,
        k_eig=cfg.dataset.k_eig,
        heads=cfg.model.heads
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = torch.nn.BCEWithLogitsLoss() # Standard for peptides-func multi-label

    # 3. Training Loop
    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    main()