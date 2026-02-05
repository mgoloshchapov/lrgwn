import hydra
from omegaconf import DictConfig
import torch
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, LaplacianLambdaMax
from sklearn.metrics import average_precision_score
import wandb
import numpy as np

from lrgwn.model import SpectralGPSModel
from lrgwn.utils import CachedSpectralTransform


def _to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


def _eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data)
            loss = criterion(logits, data.y.float())
            total_loss += loss.item()
            all_logits.append(_to_numpy(logits))
            all_targets.append(_to_numpy(data.y.float()))
    avg_loss = total_loss / max(len(loader), 1)
    if all_logits:
        logits_np = np.concatenate(all_logits, axis=0)
        targets_np = np.concatenate(all_targets, axis=0)
        ap = average_precision_score(targets_np, logits_np, average="macro")
    else:
        ap = 0.0
    return avg_loss, ap


@hydra.main(version_base=None, config_path="configs/", config_name="peptides-func")
def main(cfg: DictConfig):
    if cfg.train.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg.train.device)
    
    # 1. Load Dataset with Spectral Transforms
    transform = Compose([CachedSpectralTransform(k=cfg.dataset.k_eig), LaplacianLambdaMax(normalization='sym')])
    train_dataset = LRGBDataset(root=cfg.dataset.root, name=cfg.dataset.name, split='train', pre_transform=transform)
    val_dataset = LRGBDataset(root=cfg.dataset.root, name=cfg.dataset.name, split='val', pre_transform=transform)
    test_dataset = LRGBDataset(root=cfg.dataset.root, name=cfg.dataset.name, split='test', pre_transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size)

    # 2. Setup Model
    model = SpectralGPSModel(
        in_channels=train_dataset.num_features,
        hidden_channels=cfg.model.hidden_channels,
        out_channels=train_dataset.num_classes,
        num_layers=cfg.model.num_layers,
        K_cheb=cfg.model.K_cheb,
        k_eig=cfg.dataset.k_eig,
        heads=cfg.model.heads,
        num_scales=getattr(cfg.model, "num_scales", 1),
        num_gaussians=getattr(cfg.model, "num_gaussians", cfg.dataset.k_eig),
        shared_filters=getattr(cfg.model, "shared_filters", False),
        admissible=getattr(cfg.model, "admissible", False),
        aggregation=getattr(cfg.model, "aggregation", "concat"),
        dropout=getattr(cfg.model, "dropout", 0.0),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = torch.nn.BCEWithLogitsLoss() # Standard for peptides-func multi-label
    test_interval = int(getattr(cfg.train, "test_interval", 10))

    use_wandb = bool(getattr(cfg, "logging", {}).get("use_wandb", False))
    wandb_kwargs = {
        "project": getattr(cfg.logging, "project", None),
        "entity": getattr(cfg.logging, "entity", None),
        "name": getattr(cfg.logging, "name", None),
        "tags": getattr(cfg.logging, "tags", None),
        "config": {
            "dataset": dict(cfg.dataset),
            "model": dict(cfg.model),
            "train": dict(cfg.train),
        },
        "mode": "online" if use_wandb else "disabled",
    }
    wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
    wandb.init(**wandb_kwargs)

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
        
        train_loss = total_loss / max(len(train_loader), 1)
        val_loss, val_ap = _eval_epoch(model, val_loader, criterion, device)
        metrics = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/avg_precision": val_ap,
        }

        if (epoch + 1) % test_interval == 0:
            test_loss, test_ap = _eval_epoch(model, test_loader, criterion, device)
            metrics["test/loss"] = test_loss
            metrics["test/avg_precision"] = test_ap

        if (epoch + 1) % int(getattr(cfg.logging, "log_every", 1)) == 0:
            wandb.log(metrics)

        print(
            f"Epoch {epoch+1}, train loss {train_loss:.4f}, val loss {val_loss:.4f}, "
            f"val AP {val_ap:.4f}"
        )

    wandb.finish()

if __name__ == "__main__":
    main()
