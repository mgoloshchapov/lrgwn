from __future__ import annotations

import importlib

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import (
    AddRandomWalkPE,
    Compose,
    LaplacianLambdaMax,
)

from lrgwn.config import ExperimentConfig
from lrgwn.model import SpectralGPSModel
from lrgwn.transforms import SafeAddLaplacianEigenvectorPE
from lrgwn.utils import CachedSpectralTransform


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_cfg)


def _load_class(target: str):
    module_name, class_name = target.rsplit(".", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _compose_or_none(transforms):
    if not transforms:
        return None
    if len(transforms) == 1:
        return transforms[0]
    return Compose(transforms)


def _build_transforms(cfg: ExperimentConfig):
    base_transforms = [CachedSpectralTransform(k=cfg.dataset.k_eig)]
    if cfg.dataset.use_lambda_max:
        base_transforms.append(LaplacianLambdaMax(normalization="sym"))
    encoding_transforms = []

    pos_dim = 0
    pos_cfg = cfg.model.positional_encoding
    if pos_cfg.enabled:
        if pos_cfg.kind == "laplacian_eigenvector":
            encoding_transforms.append(
                SafeAddLaplacianEigenvectorPE(
                    k=pos_cfg.dim,
                    attr_name=pos_cfg.attr_name,
                    is_undirected=pos_cfg.is_undirected,
                )
            )
        elif pos_cfg.kind == "random_walk":
            encoding_transforms.append(
                AddRandomWalkPE(
                    walk_length=pos_cfg.dim,
                    attr_name=pos_cfg.attr_name,
                )
            )
        else:
            raise ValueError(f"Unsupported positional_encoding.kind='{pos_cfg.kind}'.")
        pos_dim = pos_cfg.dim

    struct_dim = 0
    struct_cfg = cfg.model.structural_encoding
    if struct_cfg.enabled:
        if struct_cfg.kind == "random_walk":
            encoding_transforms.append(
                AddRandomWalkPE(
                    walk_length=struct_cfg.dim,
                    attr_name=struct_cfg.attr_name,
                )
            )
        elif struct_cfg.kind == "laplacian_eigenvector":
            encoding_transforms.append(
                SafeAddLaplacianEigenvectorPE(
                    k=struct_cfg.dim,
                    attr_name=struct_cfg.attr_name,
                    is_undirected=True,
                )
            )
        else:
            raise ValueError(f"Unsupported structural_encoding.kind='{struct_cfg.kind}'.")
        struct_dim = struct_cfg.dim

    if pos_cfg.enabled and struct_cfg.enabled and pos_cfg.attr_name == struct_cfg.attr_name:
        raise ValueError(
            "positional_encoding.attr_name and structural_encoding.attr_name must differ."
        )

    if cfg.dataset.precompute_transforms:
        pre_transform = _compose_or_none(base_transforms)
        transform = _compose_or_none(encoding_transforms)
    else:
        pre_transform = None
        transform = _compose_or_none(base_transforms + encoding_transforms)

    return pre_transform, transform, pos_dim, struct_dim


def _split_lengths(total_len: int, ratios: list[float]) -> list[int]:
    if len(ratios) != 3:
        raise ValueError("dataset.random_split must contain exactly three values.")
    total_ratio = float(sum(ratios))
    if total_ratio <= 0.0:
        raise ValueError("dataset.random_split values must sum to a positive number.")

    train_len = int(total_len * (ratios[0] / total_ratio))
    val_len = int(total_len * (ratios[1] / total_ratio))
    test_len = total_len - train_len - val_len
    return [train_len, val_len, test_len]


def _build_datasets(cfg: ExperimentConfig, pre_transform, transform):
    dataset_cls = _load_class(cfg.dataset.target)
    init_args = dict(cfg.dataset.init_args)
    dataset_kwargs = {}
    if pre_transform is not None:
        dataset_kwargs["pre_transform"] = pre_transform
    if transform is not None:
        dataset_kwargs["transform"] = transform
    if cfg.dataset.force_reload:
        dataset_kwargs["force_reload"] = True

    if cfg.dataset.split_arg:
        datasets = {}
        for split_name in ("train", "val", "test"):
            if split_name not in cfg.dataset.splits:
                raise ValueError(f"dataset.splits missing '{split_name}'.")
            kwargs = dict(init_args)
            kwargs[cfg.dataset.split_arg] = cfg.dataset.splits[split_name]
            kwargs.update(dataset_kwargs)
            datasets[split_name] = dataset_cls(**kwargs)
        return datasets["train"], datasets["val"], datasets["test"]

    kwargs = dict(init_args)
    kwargs.update(dataset_kwargs)
    full_dataset = dataset_cls(**kwargs)
    lengths = _split_lengths(len(full_dataset), cfg.dataset.random_split)
    generator = torch.Generator().manual_seed(cfg.dataset.split_seed)
    train_ds, val_ds, test_ds = random_split(full_dataset, lengths, generator=generator)
    return train_ds, val_ds, test_ds


def _infer_in_channels(dataset, sample) -> int:
    num_features = getattr(dataset, "num_features", None)
    if num_features is None and hasattr(dataset, "dataset"):
        num_features = getattr(dataset.dataset, "num_features", None)

    if num_features is not None and int(num_features) > 0:
        return int(num_features)

    x = getattr(sample, "x", None)
    if x is None:
        return 1
    if x.dim() == 1:
        return 1
    return int(x.size(-1))


def _infer_out_channels(dataset, sample, task_type: str) -> int:
    if task_type == "multiclass_classification":
        num_classes = getattr(dataset, "num_classes", None)
        if num_classes is None and hasattr(dataset, "dataset"):
            num_classes = getattr(dataset.dataset, "num_classes", None)
        if num_classes is not None and int(num_classes) > 0:
            return int(num_classes)
        y = sample.y
        if y.dim() == 0:
            return int(y.item()) + 1
        return int(y.max().item()) + 1

    y = sample.y
    if y.dim() == 0:
        return 1
    return int(y.numel())


def _build_criterion(task_type: str):
    if task_type == "multilabel_classification":
        return torch.nn.BCEWithLogitsLoss()
    if task_type == "multiclass_classification":
        return torch.nn.CrossEntropyLoss()
    if task_type == "regression":
        return torch.nn.MSELoss()
    raise ValueError(f"Unsupported task.type='{task_type}'.")


def _compute_loss_and_targets(task_type: str, criterion, logits, y):
    if task_type == "multilabel_classification":
        targets = y.float()
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        targets = targets.view_as(logits)
        return criterion(logits, targets), logits, targets

    if task_type == "multiclass_classification":
        targets = y.view(-1).long()
        return criterion(logits, targets), logits, targets

    if task_type == "regression":
        targets = y.float()
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        targets = targets.view_as(logits)
        return criterion(logits, targets), logits, targets

    raise ValueError(f"Unsupported task.type='{task_type}'.")


def _compute_metric(metric_name: str, logits_list, targets_list) -> float:
    if not logits_list:
        return 0.0

    logits = torch.cat(logits_list, dim=0)
    targets = torch.cat(targets_list, dim=0)

    if metric_name == "average_precision":
        logits_np = logits.numpy()
        targets_np = targets.numpy()
        try:
            return float(average_precision_score(targets_np, logits_np, average="macro"))
        except ValueError:
            return 0.0

    if metric_name == "accuracy":
        preds = logits.argmax(dim=-1)
        return float((preds == targets).float().mean().item())

    if metric_name == "mae":
        return float(torch.mean(torch.abs(logits - targets)).item())

    raise ValueError(f"Unsupported task.metric='{metric_name}'.")


def _eval_epoch(model, loader, criterion, device, task_type: str, metric_name: str):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data)
            loss, logits_for_metric, targets = _compute_loss_and_targets(
                task_type, criterion, logits, data.y
            )
            total_loss += float(loss.item())
            all_logits.append(logits_for_metric.detach().cpu())
            all_targets.append(targets.detach().cpu())

    avg_loss = total_loss / max(len(loader), 1)
    metric_value = _compute_metric(metric_name, all_logits, all_targets)
    return avg_loss, metric_value


@hydra.main(version_base=None, config_path="configs/", config_name="peptides-func")
def main(cfg: DictConfig):
    cfg = OmegaConf.merge(OmegaConf.structured(ExperimentConfig()), cfg)
    cfg_obj: ExperimentConfig = OmegaConf.to_object(cfg)

    device = _resolve_device(cfg_obj.train.device)

    if cfg_obj.train.deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    pre_transform, transform, positional_dim, structural_dim = _build_transforms(cfg_obj)
    train_dataset, val_dataset, test_dataset = _build_datasets(
        cfg_obj, pre_transform, transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg_obj.train.batch_size,
        shuffle=True,
        num_workers=cfg_obj.train.num_workers,
        pin_memory=cfg_obj.train.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg_obj.train.batch_size,
        num_workers=cfg_obj.train.num_workers,
        pin_memory=cfg_obj.train.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg_obj.train.batch_size,
        num_workers=cfg_obj.train.num_workers,
        pin_memory=cfg_obj.train.pin_memory,
    )

    sample_graph = train_dataset[0]
    in_channels = _infer_in_channels(train_dataset, sample_graph)
    out_channels = _infer_out_channels(train_dataset, sample_graph, cfg_obj.task.type)

    model = SpectralGPSModel(
        in_channels=in_channels,
        hidden_channels=cfg_obj.model.hidden_channels,
        out_channels=out_channels,
        num_layers=cfg_obj.model.num_layers,
        K_cheb=cfg_obj.model.K_cheb,
        num_scales=cfg_obj.model.num_scales,
        num_gaussians=cfg_obj.model.num_gaussians,
        lambda_cut=cfg_obj.model.lambda_cut,
        shared_filters=cfg_obj.model.shared_filters,
        admissible=cfg_obj.model.admissible,
        aggregation=cfg_obj.model.aggregation,
        dropout=cfg_obj.model.dropout,
        positional_dim=positional_dim,
        structural_dim=structural_dim,
        positional_attr=cfg_obj.model.positional_encoding.attr_name,
        structural_attr=cfg_obj.model.structural_encoding.attr_name,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_obj.train.lr)
    criterion = _build_criterion(cfg_obj.task.type)

    use_wandb = bool(cfg_obj.logging.use_wandb)
    wandb_kwargs = {
        "project": cfg_obj.logging.project,
        "entity": cfg_obj.logging.entity,
        "name": cfg_obj.logging.name,
        "tags": cfg_obj.logging.tags,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "mode": "online" if use_wandb else "disabled",
    }
    wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
    wandb.init(**wandb_kwargs)

    for epoch in range(cfg_obj.train.epochs):
        model.train()
        total_loss = 0.0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss, _, _ = _compute_loss_and_targets(cfg_obj.task.type, criterion, logits, data.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        train_loss = total_loss / max(len(train_loader), 1)
        val_loss, val_metric = _eval_epoch(
            model,
            val_loader,
            criterion,
            device,
            task_type=cfg_obj.task.type,
            metric_name=cfg_obj.task.metric,
        )

        metrics = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/loss": val_loss,
            f"val/{cfg_obj.task.metric}": val_metric,
        }

        if cfg_obj.train.test_interval > 0 and (epoch + 1) % cfg_obj.train.test_interval == 0:
            test_loss, test_metric = _eval_epoch(
                model,
                test_loader,
                criterion,
                device,
                task_type=cfg_obj.task.type,
                metric_name=cfg_obj.task.metric,
            )
            metrics["test/loss"] = test_loss
            metrics[f"test/{cfg_obj.task.metric}"] = test_metric

        if (epoch + 1) % cfg_obj.logging.log_every == 0:
            wandb.log(metrics)

        print(
            f"Epoch {epoch + 1}, train loss {train_loss:.4f}, "
            f"val loss {val_loss:.4f}, val {cfg_obj.task.metric} {val_metric:.4f}"
        )

    wandb.finish()


if __name__ == "__main__":
    main()
