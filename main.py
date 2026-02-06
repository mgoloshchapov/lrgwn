from __future__ import annotations

import importlib

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from lrgwn.config import ExperimentConfig
from lrgwn.model import SpectralGPSModel
from lrgwn.utils import build_dataset_transforms


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cpu")
    if device_cfg == "mps":
        raise ValueError("MPS support is disabled for this project. Use train.device=cpu.")
    return torch.device(device_cfg)


def _load_class(target: str):
    module_name, class_name = target.rsplit(".", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


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
    force_reload = bool(cfg.dataset.force_reload)

    if cfg.dataset.split_arg:
        datasets = {}
        for split_idx, split_name in enumerate(("train", "val", "test")):
            if split_name not in cfg.dataset.splits:
                raise ValueError(f"dataset.splits missing '{split_name}'.")
            kwargs = dict(init_args)
            kwargs[cfg.dataset.split_arg] = cfg.dataset.splits[split_name]
            if pre_transform is not None:
                kwargs["pre_transform"] = pre_transform
            if transform is not None:
                kwargs["transform"] = transform
            # Reload once to refresh processed artifacts, then reuse for other splits.
            if force_reload and split_idx == 0:
                kwargs["force_reload"] = True
            datasets[split_name] = dataset_cls(**kwargs)
        return datasets["train"], datasets["val"], datasets["test"]

    kwargs = dict(init_args)
    if pre_transform is not None:
        kwargs["pre_transform"] = pre_transform
    if transform is not None:
        kwargs["transform"] = transform
    if force_reload:
        kwargs["force_reload"] = True
    full_dataset = dataset_cls(**kwargs)
    lengths = _split_lengths(len(full_dataset), cfg.dataset.random_split)
    generator = torch.Generator().manual_seed(cfg.dataset.split_seed)
    train_ds, val_ds, test_ds = random_split(full_dataset, lengths, generator=generator)
    return train_ds, val_ds, test_ds


def _required_encoding_attrs(cfg: ExperimentConfig) -> list[str]:
    attrs: list[str] = []
    if cfg.model.positional_encoding.enabled and cfg.model.positional_encoding.dim > 0:
        attrs.append(cfg.model.positional_encoding.attr_name)
    if cfg.model.structural_encoding.enabled and cfg.model.structural_encoding.dim > 0:
        attrs.append(cfg.model.structural_encoding.attr_name)
    return attrs


def _missing_encoding_attrs(dataset, required_attrs: list[str]) -> list[str]:
    if not required_attrs or len(dataset) == 0:
        return []
    sample = dataset[0]
    return [attr for attr in required_attrs if not hasattr(sample, attr)]


def _ensure_precomputed_encodings(
    cfg: ExperimentConfig,
    pre_transform,
    transform,
    train_dataset,
    val_dataset,
    test_dataset,
):
    required_attrs = _required_encoding_attrs(cfg)
    if not required_attrs or not cfg.dataset.precompute_transforms or pre_transform is None:
        return train_dataset, val_dataset, test_dataset

    missing = {
        "train": _missing_encoding_attrs(train_dataset, required_attrs),
        "val": _missing_encoding_attrs(val_dataset, required_attrs),
        "test": _missing_encoding_attrs(test_dataset, required_attrs),
    }
    has_missing = any(missing_attrs for missing_attrs in missing.values())
    if not has_missing:
        return train_dataset, val_dataset, test_dataset

    if cfg.dataset.force_reload:
        missing_summary = ", ".join(
            f"{split}={attrs}" for split, attrs in missing.items() if attrs
        )
        raise ValueError(
            "Required precomputed encoding attributes are missing even with "
            f"dataset.force_reload=true: {missing_summary}."
        )

    print(
        "Detected stale processed dataset cache without required encoding attributes "
        f"{required_attrs}. Rebuilding dataset once with force_reload=True."
    )
    old_force_reload = cfg.dataset.force_reload
    cfg.dataset.force_reload = True
    try:
        return _build_datasets(cfg, pre_transform, transform)
    finally:
        cfg.dataset.force_reload = old_force_reload


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
        try:
            from sklearn.metrics import average_precision_score

            logits_np = logits.numpy()
            targets_np = targets.numpy()
            return float(average_precision_score(targets_np, logits_np, average="macro"))
        except Exception:
            return _multilabel_average_precision_macro(logits, targets)

    if metric_name == "accuracy":
        preds = logits.argmax(dim=-1)
        return float((preds == targets).float().mean().item())

    if metric_name == "mae":
        return float(torch.mean(torch.abs(logits - targets)).item())

    raise ValueError(f"Unsupported task.metric='{metric_name}'.")


def _multilabel_average_precision_macro(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.dim() == 1:
        logits = logits.unsqueeze(-1)
    if targets.dim() == 1:
        targets = targets.unsqueeze(-1)

    scores = []
    targets = (targets > 0.5).to(dtype=torch.float32)
    for class_idx in range(logits.size(-1)):
        y_true = targets[:, class_idx]
        y_score = logits[:, class_idx]
        positive_count = int(y_true.sum().item())
        if positive_count == 0:
            scores.append(0.0)
            continue

        order = torch.argsort(y_score, descending=True)
        y_sorted = y_true[order]
        tp_cum = torch.cumsum(y_sorted, dim=0)
        ranks = torch.arange(1, y_sorted.numel() + 1, device=y_sorted.device)
        precision = tp_cum / ranks
        ap = float((precision * y_sorted).sum().item() / positive_count)
        scores.append(ap)

    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


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

    pre_transform, transform, positional_dim, structural_dim = build_dataset_transforms(
        cfg_obj
    )
    train_dataset, val_dataset, test_dataset = _build_datasets(
        cfg_obj, pre_transform, transform
    )
    train_dataset, val_dataset, test_dataset = _ensure_precomputed_encodings(
        cfg_obj,
        pre_transform,
        transform,
        train_dataset,
        val_dataset,
        test_dataset,
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
