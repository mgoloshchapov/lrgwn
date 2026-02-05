"""Tests for CachedSpectralTransform using hardcoded peptides-func examples.

The edge indices below are copied from data/peptides-func/raw/train.pt
(indices: 3507, 1652, 3739, 1440, 5553). We avoid loading the full
LRGB dataset at test time.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix

from lrgwn.utils import CachedSpectralTransform

EXAMPLES: List[Dict[str, object]] = [
    {
        "num_nodes": 8,
        "edge_index": [
            [0, 1, 1, 2, 2, 3, 2, 4, 4, 5, 5, 6, 6, 7],
            [1, 0, 2, 1, 3, 2, 4, 2, 5, 4, 6, 5, 7, 6],
        ],
    },
    {
        "num_nodes": 9,
        "edge_index": [
            [0, 1, 1, 2, 1, 3, 3, 4, 3, 5, 5, 6, 6, 7, 7, 8],
            [1, 0, 2, 1, 3, 1, 4, 3, 5, 3, 6, 5, 7, 6, 8, 7],
        ],
    },
    {
        "num_nodes": 9,
        "edge_index": [
            [0, 1, 1, 2, 2, 3, 3, 4, 1, 5, 5, 6, 5, 7, 7, 8],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 1, 6, 5, 7, 5, 8, 7],
        ],
    },
    {
        "num_nodes": 10,
        "edge_index": [
            [1, 2, 3, 4, 4, 5, 4, 6, 6, 7],
            [2, 1, 4, 3, 5, 4, 6, 4, 7, 6],
        ],
    },
    {
        "num_nodes": 10,
        "edge_index": [
            [0, 1, 1, 2, 2, 3, 2, 4, 4, 5, 5, 6, 6, 7, 5, 8, 8, 9],
            [1, 0, 2, 1, 3, 2, 4, 2, 5, 4, 6, 5, 7, 6, 8, 5, 9, 8],
        ],
    },
]


def _laplacian_dense(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index, edge_weight = get_laplacian(
        edge_index, normalization="sym", num_nodes=num_nodes
    )
    laplacian = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
    return torch.from_numpy(laplacian.toarray()).float()


def test_cached_spectral_transform_eigenpairs() -> None:
    transform = CachedSpectralTransform(k=20)

    for example in EXAMPLES:
        edge_index = torch.tensor(example["edge_index"], dtype=torch.long)
        num_nodes = int(example["num_nodes"])
        data = Data(edge_index=edge_index, num_nodes=num_nodes)

        out = transform(data)
        k = int(transform.k)
        k_eff = min(k, num_nodes - 1)
        Lambda = out.Lambda.squeeze(0)

        assert out.Lambda.shape == (1, k)
        assert out.U.shape == (num_nodes, k)
        assert torch.isfinite(out.Lambda).all()
        assert torch.isfinite(out.U).all()

        if k_eff == 0:
            continue

        U_eff = out.U[:, :k_eff]
        gram = U_eff.t() @ U_eff
        torch.testing.assert_close(
            gram, torch.eye(k_eff), rtol=1e-4, atol=1e-4
        )

        laplacian = _laplacian_dense(edge_index, num_nodes)
        residual = laplacian @ U_eff - U_eff * Lambda[:k_eff]
        assert residual.abs().max().item() < 1e-3


def test_cached_spectral_transform_empty_edges() -> None:
    transform = CachedSpectralTransform(k=5)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    data = Data(edge_index=edge_index, num_nodes=4)

    out = transform(data)

    assert out.Lambda.shape == (1, 5)
    assert torch.allclose(out.Lambda.squeeze(0), torch.zeros(5))
    assert out.U.shape == (4, 5)
    torch.testing.assert_close(out.U[:, :3], torch.eye(4)[:, :3], rtol=0, atol=0)
    assert torch.allclose(out.U[:, 3:], torch.zeros((4, 2)))


def test_cached_spectral_transform_num_nodes_inferred_from_x() -> None:
    transform = CachedSpectralTransform(k=5)
    x = torch.randn(3, 2)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)

    out = transform(data)

    assert out.U.shape == (3, 5)
    assert out.Lambda.shape == (1, 5)


def test_cached_spectral_transform_k_eff_zero() -> None:
    transform = CachedSpectralTransform(k=5)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    data = Data(edge_index=edge_index, num_nodes=1)

    out = transform(data)

    assert out.Lambda.shape == (1, 5)
    assert out.U.shape == (1, 5)
    assert torch.allclose(out.Lambda.squeeze(0), torch.zeros(5))
    assert torch.allclose(out.U, torch.zeros((1, 5)))


def test_cached_spectral_transform_zero_nodes() -> None:
    transform = CachedSpectralTransform(k=5)
    data = Data(num_nodes=0)

    out = transform(data)

    assert out.Lambda.shape == (1, 5)
    assert out.U.shape == (0, 5)
    assert torch.allclose(out.Lambda.squeeze(0), torch.zeros(5))
