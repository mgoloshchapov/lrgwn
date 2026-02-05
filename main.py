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


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_cfg)


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
        try:
            ap = average_precision_score(targets_np, logits_np, average="macro")
        except ValueError:
            ap = 0.0
    else:
        ap = 0.0
    return avg_loss, ap


@hydra.main(version_base=None, config_path="configs/", config_name="peptides-func")
def main(cfg: DictConfig):
    device = _resolve_device(cfg.train.device)

    if cfg.train.deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    transform = Compose(
        [
            CachedSpectralTransform(k=cfg.dataset.k_eig),
            LaplacianLambdaMax(normalization="sym"),
        ]
    )
    train_dataset = LRGBDataset(
        root=cfg.dataset.root,
        name=cfg.dataset.name,
        split="train",
        pre_transform=transform,
    )
    val_dataset = LRGBDataset(
        root=cfg.dataset.root,
        name=cfg.dataset.name,
        split="val",
        pre_transform=transform,
    )
    test_dataset = LRGBDataset(
        root=cfg.dataset.root,
        name=cfg.dataset.name,
        split="test",
        pre_transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
    )

    model = SpectralGPSModel(
        in_channels=train_dataset.num_features,
        hidden_channels=cfg.model.hidden_channels,
        out_channels=train_dataset.num_classes,
        num_layers=cfg.model.num_layers,
        K_cheb=cfg.model.K_cheb,
        num_scales=cfg.model.num_scales,
        num_gaussians=cfg.model.num_gaussians,
        lambda_cut=cfg.model.lambda_cut,
        shared_filters=cfg.model.shared_filters,
        admissible=cfg.model.admissible,
        aggregation=cfg.model.aggregation,
        dropout=cfg.model.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    use_wandb = bool(cfg.logging.use_wandb)
    wandb_kwargs = {
        "project": cfg.logging.project,
        "entity": cfg.logging.entity,
        "name": cfg.logging.name,
        "tags": cfg.logging.tags,
        "config": {
            "dataset": dict(cfg.dataset),
            "model": dict(cfg.model),
            "train": dict(cfg.train),
        },
        "mode": "online" if use_wandb else "disabled",
    }
    wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
    wandb.init(**wandb_kwargs)

    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0.0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, data.y.float())
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

        if cfg.train.test_interval > 0 and (epoch + 1) % cfg.train.test_interval == 0:
            test_loss, test_ap = _eval_epoch(model, test_loader, criterion, device)
            metrics["test/loss"] = test_loss
            metrics["test/avg_precision"] = test_ap

        if (epoch + 1) % cfg.logging.log_every == 0:
            wandb.log(metrics)

        print(
            f"Epoch {epoch + 1}, train loss {train_loss:.4f}, "
            f"val loss {val_loss:.4f}, val AP {val_ap:.4f}"
        )

    wandb.finish()


if __name__ == "__main__":
    main()
