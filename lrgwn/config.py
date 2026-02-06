from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetConfig:
    target: str = "torch_geometric.datasets.LRGBDataset"
    init_args: dict[str, Any] = field(
        default_factory=lambda: {"root": "./data", "name": "Peptides-func"}
    )
    split_arg: str | None = "split"
    splits: dict[str, Any] = field(
        default_factory=lambda: {"train": "train", "val": "val", "test": "test"}
    )
    random_split: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    split_seed: int = 42
    k_eig: int = 20
    precompute_transforms: bool = True
    use_lambda_max: bool = True
    force_reload: bool = False


@dataclass
class PositionalEncodingConfig:
    enabled: bool = True
    kind: str = "laplacian_eigenvector"
    dim: int = 16
    attr_name: str = "pe"
    is_undirected: bool = True


@dataclass
class StructuralEncodingConfig:
    enabled: bool = True
    kind: str = "random_walk"
    dim: int = 16
    attr_name: str = "se"

@dataclass
class ModelConfig:
    hidden_channels: int = 128
    num_layers: int = 4
    K_cheb: int = 3
    num_scales: int = 1
    num_gaussians: int = 16
    lambda_cut: float = 2.0
    shared_filters: bool = False
    admissible: bool = False
    aggregation: str = "concat"
    dropout: float = 0.0
    positional_encoding: PositionalEncodingConfig = field(
        default_factory=PositionalEncodingConfig
    )
    structural_encoding: StructuralEncodingConfig = field(
        default_factory=StructuralEncodingConfig
    )


@dataclass
class TaskConfig:
    type: str = "multilabel_classification"
    metric: str = "average_precision"


@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 32
    device: str = "auto"
    test_interval: int = 10
    num_workers: int = 0
    pin_memory: bool = False
    deterministic: bool = False


@dataclass
class LoggingConfig:
    use_wandb: bool = True
    project: str = "lrgwn"
    entity: str | None = None
    name: str | None = None
    tags: list[str] = field(default_factory=list)
    log_every: int = 1


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
