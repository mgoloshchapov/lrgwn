from __future__ import annotations

import hashlib
import numpy as np
import scipy
import torch
from pathlib import Path
from scipy.sparse.linalg import eigsh
from torch_geometric.transforms import (
    AddLaplacianEigenvectorPE,
    AddRandomWalkPE,
    BaseTransform,
    Compose,
    LaplacianLambdaMax,
)
from torch_geometric.utils import (
    get_laplacian,
    is_undirected,
    scatter,
    to_scipy_sparse_matrix,
    to_undirected,
)


def _infer_num_nodes(data) -> int:
    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is not None:
        return int(num_nodes)
    x = getattr(data, "x", None)
    if x is not None:
        return int(x.size(0))
    edge_index = getattr(data, "edge_index", None)
    if edge_index is not None and edge_index.numel() > 0:
        return int(edge_index.max().item()) + 1
    return 0


class CachedSpectralTransform(BaseTransform):
    def __init__(
        self,
        k: int,
        dense: bool = False,
        which: str = "LM",
        sigma: float | None = 1e-6,
        cache_dir: str | None = None,
        memory_cache: bool = True,
    ):
        self.k = int(k)
        self.dense = bool(dense)
        self.which = str(which).upper()
        self.sigma = None if sigma is None else float(sigma)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache_enabled = bool(memory_cache)
        self._memory_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _has_expected_attrs(self, data) -> bool:
        if not (hasattr(data, "Lambda") and hasattr(data, "U") and hasattr(data, "Lambda_mask")):
            return False
        try:
            lambda_dim = int(data.Lambda.size(-1))
            u_dim = int(data.U.size(-1))
            mask_dim = int(data.Lambda_mask.size(-1))
        except Exception:
            return False
        return lambda_dim == self.k and u_dim == self.k and mask_dim == self.k

    def _graph_cache_key(self, data, num_nodes: int) -> str:
        hasher = hashlib.sha1()
        hasher.update(np.asarray([num_nodes], dtype=np.int64).tobytes())
        edge_index = getattr(data, "edge_index", None)
        if edge_index is not None:
            edge_index_np = (
                edge_index.detach().cpu().to(torch.int64).contiguous().numpy()
            )
            hasher.update(edge_index_np.tobytes())
        edge_weight = getattr(data, "edge_weight", None)
        if edge_weight is not None:
            edge_weight_np = (
                edge_weight.detach().cpu().to(torch.float32).contiguous().numpy()
            )
            hasher.update(edge_weight_np.tobytes())
        return hasher.hexdigest()

    def _assign_cached(self, data, cached):
        Lambda, U, Lambda_mask = cached
        data.Lambda = Lambda.clone()
        data.U = U.clone()
        data.Lambda_mask = Lambda_mask.clone()
        return data

    def _load_from_disk(self, key: str):
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.pt"
        if not path.exists():
            return None
        payload = torch.load(path, map_location="cpu")
        Lambda = payload.get("Lambda")
        U = payload.get("U")
        Lambda_mask = payload.get("Lambda_mask")
        if Lambda is None or U is None or Lambda_mask is None:
            return None
        if int(Lambda.size(-1)) != self.k or int(U.size(-1)) != self.k or int(Lambda_mask.size(-1)) != self.k:
            return None
        return Lambda, U, Lambda_mask

    def _save_to_disk(self, key: str, payload) -> None:
        if self.cache_dir is None:
            return
        path = self.cache_dir / f"{key}.pt"
        if path.exists():
            return
        tmp_path = self.cache_dir / f"{key}.tmp"
        torch.save(
            {
                "Lambda": payload[0],
                "U": payload[1],
                "Lambda_mask": payload[2],
            },
            tmp_path,
        )
        tmp_path.replace(path)

    def _build_laplacian(self, data, num_nodes: int):
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        edge_weight = getattr(data, "edge_weight", None)
        if edge_weight is None and edge_index.numel() > 0:
            edge_weight = torch.ones(
                edge_index.size(1), dtype=torch.float32, device=edge_index.device
            )

        if edge_index.numel() > 0 and not is_undirected(edge_index, edge_weight):
            edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce="mean")

        lap_edge_index, lap_edge_weight = get_laplacian(
            edge_index,
            edge_weight,
            normalization="sym",
            num_nodes=num_nodes,
        )
        return to_scipy_sparse_matrix(lap_edge_index, lap_edge_weight, num_nodes)

    def _compute_eig(self, L, k_eff: int, num_nodes: int):
        if self.dense or k_eff >= num_nodes:
            eig_vals, eig_vecs = scipy.linalg.eigh(L.toarray())
            order = np.argsort(eig_vals)
            order = order[:k_eff]
            eig_vals = eig_vals[order]
            eig_vecs = eig_vecs[:, order]
            return eig_vals, eig_vecs

        kwargs = {}
        if self.sigma is not None:
            kwargs["sigma"] = self.sigma
        eig_vals, eig_vecs = eigsh(
            L,
            k=k_eff,
            which=self.which,
            return_eigenvectors=True,
            **kwargs,
        )
        order = eig_vals.argsort()[: min(k_eff, len(eig_vals))]
        eig_vals = eig_vals[order]
        eig_vecs = eig_vecs[:, order]
        return eig_vals, eig_vecs

    def forward(self, data):
        if self._has_expected_attrs(data):
            return data

        num_nodes = _infer_num_nodes(data)
        k = max(self.k, 0)
        if num_nodes <= 0 or k == 0:
            data.Lambda = torch.zeros((1, k), dtype=torch.float32)
            data.U = torch.zeros((max(num_nodes, 0), k), dtype=torch.float32)
            data.Lambda_mask = torch.zeros((k,), dtype=torch.bool)
            return data

        key = None
        if self.memory_cache_enabled or self.cache_dir is not None:
            key = self._graph_cache_key(data, num_nodes)
            if self.memory_cache_enabled and key in self._memory_cache:
                return self._assign_cached(data, self._memory_cache[key])
            disk_payload = self._load_from_disk(key)
            if disk_payload is not None:
                if self.memory_cache_enabled:
                    self._memory_cache[key] = disk_payload
                return self._assign_cached(data, disk_payload)

        k_eff = min(k, num_nodes)
        eig_mask = np.zeros(k, dtype=bool)
        eig_mask[:k_eff] = True

        L = self._build_laplacian(data, num_nodes)
        try:
            eig_vals, eig_vecs = self._compute_eig(L, k_eff=k_eff, num_nodes=num_nodes)
        except Exception:
            eig_vals, eig_vecs = scipy.linalg.eigh(L.toarray())
            eig_vals = eig_vals[:k_eff]
            eig_vecs = eig_vecs[:, :k_eff]

        padded_vals = np.zeros(k, dtype=np.float32)
        padded_vecs = np.zeros((num_nodes, k), dtype=np.float32)
        if k_eff > 0:
            padded_vals[:k_eff] = eig_vals[:k_eff]
            padded_vecs[:, :k_eff] = eig_vecs[:, :k_eff]

        data.Lambda = torch.from_numpy(padded_vals).unsqueeze(0)
        data.U = torch.from_numpy(padded_vecs)
        data.Lambda_mask = torch.from_numpy(eig_mask)

        cached_payload = (
            data.Lambda.detach().cpu(),
            data.U.detach().cpu(),
            data.Lambda_mask.detach().cpu(),
        )
        if key is not None:
            if self.memory_cache_enabled:
                self._memory_cache[key] = cached_payload
            self._save_to_disk(key, cached_payload)

        return data


class CachedAttributeTransform(BaseTransform):
    def __init__(
        self,
        transform: BaseTransform,
        attrs: dict[str, int],
        cache_tag: str,
        cache_dir: str | None = None,
        memory_cache: bool = True,
    ):
        self.transform = transform
        self.attrs = {name: int(dim) for name, dim in attrs.items()}
        self.cache_tag = str(cache_tag)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache_enabled = bool(memory_cache)
        self._memory_cache: dict[str, dict[str, torch.Tensor]] = {}

    def _attr_dim(self, values) -> int:
        if hasattr(values, "dim"):
            return int(values.size(-1)) if values.dim() > 0 else 1
        return 1

    def _has_expected_attrs(self, data) -> bool:
        for attr, expected_dim in self.attrs.items():
            if not hasattr(data, attr):
                return False
            values = getattr(data, attr)
            if self._attr_dim(values) != expected_dim:
                return False
        return True

    def _graph_cache_key(self, data, num_nodes: int) -> str:
        hasher = hashlib.sha1()
        hasher.update(np.asarray([num_nodes], dtype=np.int64).tobytes())
        edge_index = getattr(data, "edge_index", None)
        if edge_index is not None:
            edge_index_np = (
                edge_index.detach().cpu().to(torch.int64).contiguous().numpy()
            )
            hasher.update(edge_index_np.tobytes())
        edge_weight = getattr(data, "edge_weight", None)
        if edge_weight is not None:
            edge_weight_np = (
                edge_weight.detach().cpu().to(torch.float32).contiguous().numpy()
            )
            hasher.update(edge_weight_np.tobytes())
        hasher.update(self.cache_tag.encode("utf-8"))
        return hasher.hexdigest()

    def _assign_cached(self, data, payload: dict[str, torch.Tensor]):
        for attr, values in payload.items():
            data[attr] = values.clone()
        return data

    def _load_from_disk(self, key: str):
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.pt"
        if not path.exists():
            return None
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            return None
        for attr, expected_dim in self.attrs.items():
            values = payload.get(attr)
            if values is None:
                return None
            if self._attr_dim(values) != expected_dim:
                return None
        return payload

    def _save_to_disk(self, key: str, payload: dict[str, torch.Tensor]) -> None:
        if self.cache_dir is None:
            return
        path = self.cache_dir / f"{key}.pt"
        if path.exists():
            return
        tmp_path = self.cache_dir / f"{key}.tmp"
        torch.save(payload, tmp_path)
        tmp_path.replace(path)

    def forward(self, data):
        if self._has_expected_attrs(data):
            return data

        num_nodes = _infer_num_nodes(data)
        key = None
        if self.memory_cache_enabled or self.cache_dir is not None:
            key = self._graph_cache_key(data, num_nodes)
            if self.memory_cache_enabled and key in self._memory_cache:
                return self._assign_cached(data, self._memory_cache[key])
            disk_payload = self._load_from_disk(key)
            if disk_payload is not None:
                if self.memory_cache_enabled:
                    self._memory_cache[key] = disk_payload
                return self._assign_cached(data, disk_payload)

        data = self.transform(data)
        payload: dict[str, torch.Tensor] = {}
        for attr, expected_dim in self.attrs.items():
            if not hasattr(data, attr):
                raise ValueError(
                    f"CachedAttributeTransform expected attr '{attr}' from "
                    f"{self.transform.__class__.__name__}, but it was not found."
                )
            values = getattr(data, attr)
            actual_dim = self._attr_dim(values)
            if actual_dim != expected_dim:
                raise ValueError(
                    f"CachedAttributeTransform attr '{attr}' has dim {actual_dim}, "
                    f"expected {expected_dim}."
                )
            payload[attr] = values.detach().cpu()

        if key is not None:
            if self.memory_cache_enabled:
                self._memory_cache[key] = payload
            self._save_to_disk(key, payload)

        return data


class MagLaplacianPlainPositionalEncoding(BaseTransform):
    """S2GNN-compatible positional stats from plain Laplacian eigendecomposition."""

    def __init__(
        self,
        k: int,
        attr_name: str = "laplacian_eigenvector_plain_posenc",
        sigma: float = 1e-7,
    ) -> None:
        self.k = int(k)
        self.attr_name = attr_name
        self.sigma = float(sigma)

    def _zeros(self, data, num_nodes: int) -> torch.Tensor:
        edge_index = getattr(data, "edge_index", None)
        device = edge_index.device if edge_index is not None else None
        return torch.zeros((num_nodes, self.k), dtype=torch.float32, device=device)

    def forward(self, data):
        num_nodes = _infer_num_nodes(data)
        if num_nodes <= 0 or self.k <= 0:
            data[self.attr_name] = self._zeros(data, max(num_nodes, 0))
            return data

        if not hasattr(data, "U") or not hasattr(data, "Lambda"):
            raise ValueError(
                "MagLaplacianPlainPositionalEncoding requires precomputed 'U' and 'Lambda'."
            )

        U = data.U.float()
        Lambda = data.Lambda.float()
        if Lambda.dim() == 2:
            Lambda = Lambda.squeeze(0)
        if Lambda.dim() != 1:
            raise ValueError(f"Expected Lambda to be 1D/2D tensor, got shape {data.Lambda.shape}.")

        if U.size(0) != num_nodes:
            raise ValueError(
                f"U has {U.size(0)} rows but graph has {num_nodes} nodes."
            )

        if U.size(-1) < self.k:
            U = torch.cat([U, U.new_zeros((num_nodes, self.k - U.size(-1)))], dim=-1)
        elif U.size(-1) > self.k:
            U = U[:, : self.k]

        if Lambda.numel() < self.k:
            Lambda = torch.cat([Lambda, Lambda.new_zeros((self.k - Lambda.numel(),))], dim=0)
        elif Lambda.numel() > self.k:
            Lambda = Lambda[: self.k]

        edge_index = getattr(data, "edge_index", None)
        if edge_index is None or edge_index.numel() == 0:
            data[self.attr_name] = self._zeros(data, num_nodes)
            return data

        edge_weight = getattr(data, "edge_weight", None)
        if not is_undirected(edge_index, edge_weight):
            edge_index, _ = to_undirected(edge_index, edge_weight, reduce="mean")
        row, col = edge_index

        sigma = max(self.sigma, 1e-12)
        eig_diff = Lambda.unsqueeze(0) - Lambda.unsqueeze(1)
        filter_ = torch.softmax(-((eig_diff * eig_diff) / sigma), dim=-1)
        prod_edge = U[row] * U[col]
        single_frequency_convs = torch.matmul(prod_edge, filter_)
        posenc = scatter(
            single_frequency_convs,
            row.clone().detach(),
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        ).float()

        mask = getattr(data, "Lambda_mask", None)
        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=posenc.device)
            if mask.numel() < self.k:
                pad = torch.zeros((self.k - mask.numel(),), dtype=torch.bool, device=mask.device)
                mask = torch.cat([mask, pad], dim=0)
            elif mask.numel() > self.k:
                mask = mask[: self.k]
            posenc[:, ~mask] = 0.0

        data[self.attr_name] = posenc
        return data


class SafeAddLaplacianEigenvectorPE(BaseTransform):
    """Adds Laplacian PE and pads/truncates it to a fixed dimension."""

    def __init__(
        self,
        k: int,
        attr_name: str = "laplacian_eigenvector_pe",
        is_undirected: bool = False,
    ) -> None:
        self.k = int(k)
        self.attr_name = attr_name
        self.is_undirected = is_undirected

    def _zeros(self, data, num_nodes: int) -> torch.Tensor:
        edge_index = getattr(data, "edge_index", None)
        device = edge_index.device if edge_index is not None else None
        return torch.zeros((num_nodes, self.k), dtype=torch.float32, device=device)

    def _dense_fallback_pe(self, data, num_nodes: int, k_eff: int) -> torch.Tensor:
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None or edge_index.numel() == 0:
            return self._zeros(data, num_nodes)

        edge_weight = getattr(data, "edge_weight", None)
        lap_edge_index, lap_edge_weight = get_laplacian(
            edge_index, edge_weight, normalization="sym", num_nodes=num_nodes
        )
        L = to_scipy_sparse_matrix(lap_edge_index, lap_edge_weight, num_nodes=num_nodes)
        eigvals, eigvecs = torch.linalg.eigh(torch.from_numpy(L.toarray()).float())
        pe = eigvecs[:, 1 : k_eff + 1]
        if pe.size(-1) < self.k:
            pe = torch.cat([pe, pe.new_zeros((pe.size(0), self.k - pe.size(-1)))], dim=-1)
        elif pe.size(-1) > self.k:
            pe = pe[:, : self.k]
        return pe

    def forward(self, data):
        num_nodes = _infer_num_nodes(data)
        if num_nodes <= 0 or self.k <= 0:
            data[self.attr_name] = self._zeros(data, max(num_nodes, 0))
            return data

        k_eff = min(self.k, max(num_nodes - 1, 0))
        if k_eff == 0 or getattr(data, "edge_index", None) is None:
            data[self.attr_name] = self._zeros(data, num_nodes)
            return data

        transform = AddLaplacianEigenvectorPE(
            k=k_eff,
            attr_name=self.attr_name,
            is_undirected=self.is_undirected,
        )
        try:
            data = transform(data)
            pe = data[self.attr_name]
        except (TypeError, RuntimeError, ValueError) as exc:
            # PyG may fail on tiny graphs when sparse eigsh enters the k >= N branch.
            message = str(exc)
            if "k >= N" not in message and "Cannot use scipy.linalg.eigh for sparse A" not in message:
                raise
            pe = self._dense_fallback_pe(data, num_nodes=num_nodes, k_eff=k_eff)

        if pe.size(-1) < self.k:
            pe = torch.cat(
                [pe, pe.new_zeros((pe.size(0), self.k - pe.size(-1)))],
                dim=-1,
            )
        elif pe.size(-1) > self.k:
            pe = pe[:, : self.k]

        data[self.attr_name] = pe
        return data


def _compose_or_none(transforms):
    if not transforms:
        return None
    if len(transforms) == 1:
        return transforms[0]
    return Compose(transforms)


def build_dataset_transforms(cfg):
    root = Path(cfg.dataset.init_args.get("root", "./data"))
    dataset_name = str(cfg.dataset.init_args.get("name", "dataset")).lower()
    dataset_name = dataset_name.replace(" ", "-").replace("/", "_")

    spectral_cache_dir = None
    if cfg.dataset.spectral_cache_enabled:
        if cfg.dataset.spectral_cache_dir:
            spectral_cache_dir = Path(cfg.dataset.spectral_cache_dir)
        else:
            sigma_tag = (
                "none"
                if cfg.dataset.eig_sigma is None
                else str(cfg.dataset.eig_sigma).replace(".", "p")
            )
            spectral_cache_dir = (
                root
                / dataset_name
                / "spectral_cache"
                / f"k{cfg.dataset.k_eig}_dense{int(cfg.dataset.eig_dense)}_which{cfg.dataset.eig_which}_sigma{sigma_tag}"
            )

    encoding_cache_root = None
    if cfg.dataset.spectral_cache_enabled:
        encoding_cache_root = root / dataset_name / "encoding_cache"

    precompute_transforms = bool(cfg.dataset.precompute_transforms)

    base_transforms = [
        CachedSpectralTransform(
            k=cfg.dataset.k_eig,
            dense=cfg.dataset.eig_dense,
            which=cfg.dataset.eig_which,
            sigma=cfg.dataset.eig_sigma,
            cache_dir=str(spectral_cache_dir) if spectral_cache_dir is not None else None,
            memory_cache=cfg.dataset.spectral_memory_cache,
        )
    ]
    if cfg.dataset.use_lambda_max:
        lambda_max_transform = LaplacianLambdaMax(normalization="sym")
        if not precompute_transforms and encoding_cache_root is not None:
            lambda_cache_dir = encoding_cache_root / "lambda_max_sym"
            lambda_max_transform = CachedAttributeTransform(
                transform=lambda_max_transform,
                attrs={"lambda_max": 1},
                cache_tag="lambda_max_sym",
                cache_dir=str(lambda_cache_dir),
                memory_cache=cfg.dataset.spectral_memory_cache,
            )
        base_transforms.append(lambda_max_transform)
    encoding_transforms = []

    pos_dim = 0
    pos_cfg = cfg.model.positional_encoding
    if pos_cfg.enabled:
        pos_transform = None
        if pos_cfg.kind == "laplacian_eigenvector":
            pos_transform = SafeAddLaplacianEigenvectorPE(
                k=pos_cfg.dim,
                attr_name=pos_cfg.attr_name,
                is_undirected=pos_cfg.is_undirected,
            )
        elif pos_cfg.kind == "random_walk":
            pos_transform = AddRandomWalkPE(
                walk_length=pos_cfg.dim,
                attr_name=pos_cfg.attr_name,
            )
        elif pos_cfg.kind == "maglap_positional":
            pos_transform = MagLaplacianPlainPositionalEncoding(
                k=pos_cfg.dim,
                attr_name=pos_cfg.attr_name,
                sigma=pos_cfg.sigma,
            )
        else:
            raise ValueError(f"Unsupported positional_encoding.kind='{pos_cfg.kind}'.")

        if (
            not precompute_transforms
            and encoding_cache_root is not None
            and pos_transform is not None
        ):
            pos_sigma_tag = (
                str(pos_cfg.sigma).replace(".", "p")
                if hasattr(pos_cfg, "sigma")
                else "na"
            )
            pos_cache_dir = (
                encoding_cache_root
                / f"{pos_cfg.attr_name}_{pos_cfg.kind}_d{pos_cfg.dim}_sigma{pos_sigma_tag}"
            )
            pos_transform = CachedAttributeTransform(
                transform=pos_transform,
                attrs={pos_cfg.attr_name: pos_cfg.dim},
                cache_tag=f"pos_{pos_cfg.attr_name}_{pos_cfg.kind}_d{pos_cfg.dim}_sigma{pos_sigma_tag}",
                cache_dir=str(pos_cache_dir),
                memory_cache=cfg.dataset.spectral_memory_cache,
            )

        encoding_transforms.append(pos_transform)
        pos_dim = pos_cfg.dim

    struct_dim = 0
    struct_cfg = cfg.model.structural_encoding
    if struct_cfg.enabled:
        struct_transform = None
        if struct_cfg.kind == "random_walk":
            struct_transform = AddRandomWalkPE(
                walk_length=struct_cfg.dim,
                attr_name=struct_cfg.attr_name,
            )
        elif struct_cfg.kind == "laplacian_eigenvector":
            struct_transform = SafeAddLaplacianEigenvectorPE(
                k=struct_cfg.dim,
                attr_name=struct_cfg.attr_name,
                is_undirected=True,
            )
        else:
            raise ValueError(f"Unsupported structural_encoding.kind='{struct_cfg.kind}'.")

        if (
            not precompute_transforms
            and encoding_cache_root is not None
            and struct_transform is not None
        ):
            struct_cache_dir = (
                encoding_cache_root
                / f"{struct_cfg.attr_name}_{struct_cfg.kind}_d{struct_cfg.dim}"
            )
            struct_transform = CachedAttributeTransform(
                transform=struct_transform,
                attrs={struct_cfg.attr_name: struct_cfg.dim},
                cache_tag=f"struct_{struct_cfg.attr_name}_{struct_cfg.kind}_d{struct_cfg.dim}",
                cache_dir=str(struct_cache_dir),
                memory_cache=cfg.dataset.spectral_memory_cache,
            )

        encoding_transforms.append(struct_transform)
        struct_dim = struct_cfg.dim

    if pos_cfg.enabled and struct_cfg.enabled and pos_cfg.attr_name == struct_cfg.attr_name:
        raise ValueError(
            "positional_encoding.attr_name and structural_encoding.attr_name must differ."
        )

    if precompute_transforms:
        pre_transform = _compose_or_none(base_transforms + encoding_transforms)
        transform = None
    else:
        pre_transform = None
        transform = _compose_or_none(base_transforms + encoding_transforms)

    return pre_transform, transform, pos_dim, struct_dim
