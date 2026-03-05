# Peptides-func: S2GNN Configuration Trace

This note maps `lr_gwn/configs/peptides-func/peptides-func-s2gnn.yaml` to the actual code path and summarizes the network used for `peptides-functional`.

## 1) Where the YAML is consumed

1. Entry point: `lr_gwn/main.py` loads the YAML into GraphGym `cfg` via `cfg.merge_from_file(...)`, then builds loaders/model/optimizer/scheduler.
2. Dataset selection: `dataset.format=OGB` + `dataset.name=peptides-functional` is dispatched in `graphgps/loader/master_loader.py` to `preformat_Peptides(...)`.
3. Positional/statistical preprocessing:
   - `posenc_MagLapPE.enable=True`
   - `posenc_RWSE.enable=True`
   are collected in `master_loader.py`, then `compute_posenc_stats(...)` is run on every graph.
4. Model build: `model.type=s2gnn` instantiates `graphgps/network/s2gnn.py::S2GNN`.

## 2) How initial node embeddings are extracted

### 2.1 Raw graph features (dataset loader)

`graphgps/loader/dataset/peptides_functional.py` converts each SMILES string to an OGB graph:

- `data.x`: atom categorical features (`int64`)
- `data.edge_attr`: bond categorical features (`int64`)
- `data.y`: 10-label multilabel targets

### 2.2 Encoder composition: `Atom+MagLapPE+RWSE`

The encoder name is resolved in `graphgps/encoder/composed_encoders.py` through `Concat3NodeEncoder`:

- Base atom encoder: `AtomEncoder`
- PE encoder 1: `MagLapPENodeEncoder`
- PE encoder 2: `RWSENodeEncoder`

With `gnn.dim_inner=224`, `posenc_MagLapPE.dim_pe=8`, `posenc_RWSE.dim_pe=28`:

- Atom branch output dim = `224 - 8 - 28 = 188`
- MagLapPE branch output dim = `8`
- RWSE branch output dim = `28`
- Concatenated node embedding = `224`

### 2.3 MagLapPE features

`compute_posenc_stats(...)` calls `AddMagneticLaplacianEigenvectorPlain` with this config:

- `k=150` (`posenc_MagLapPE.eigen.max_freqs`)
- `q=0.0` (real Laplacian)
- `positional_encoding=True`
- `sparse=False`
- `largest_connected_component=False`

This stores:

- `laplacian_eigenvector_plain`
- `laplacian_eigenvalue_plain`
- `laplacian_eigenvector_plain_posenc`

Then `MagLapPENodeEncoder` applies a linear projection from spectral stats to `dim_pe=8` and concatenates to node embeddings.

### 2.4 RWSE features

`compute_posenc_stats(...)` evaluates `times_func=range(1,21)` and computes 20 random-walk landing probabilities as `pestat_RWSE`.

`RWSENodeEncoder` then:

- applies `BatchNorm` to raw RW statistics (`raw_norm_type=BatchNorm`)
- applies a `Linear(20 -> 28)` PE encoder (`model=Linear`, `dim_pe=28`)
- concatenates to node embeddings

## 3) Network architecture for this YAML

Model class: `graphgps/network/s2gnn.py::S2GNN`

High-level stack:

1. Feature encoder (`Atom+MagLapPE+RWSE`) -> node states of size 224.
2. Message passing with `gnn.layers_mp=3`:
   - Spatial layer per block: `GCNConvGNNLayer` (`layer_type=gcnconv`)
   - Spectral layer per block: `FeatureBatchSpectralLayer` (`spec_layer_type=default`)
   - Activation: GELU, dropout: 0.25, node residuals enabled
3. Graph readout: mean pooling (`model.graph_pooling=mean`)
4. Prediction head: `mlp_graph` with `layers_post_mp=3`

Important detail on skips:

- `spec_layer_skip=[3]` does not skip anything when `layers_mp=3`, because skip indices are filtered by `i < layers_mp`.
- Effective spectral skips are empty in this run.

### Spectral filter details (from `posenc_MagLapPE`)

- Filter encoder: `basis`
- Gaussian basis count: `basis_num_gaussians=60`
- Bottleneck factor: `basis_bottleneck=0.2`
- Frequency cutoff: `0.7`
- Window: `tukey`
- Feature transform: `spec_feature_transform=glu_0.05`

## 4) Reproduction in `lrgwn`

`lrgwn` has a different model family (`LRGWNLayer`, Chebyshev + low-rank spectral correction), so reproduction is approximate. A mapped runnable config was added:

- `lrgwn/configs/peptides-func-s2gnn.yaml`

Main mapping:

- hidden size `224`
- layers `3`
- dropout `0.25`
- encoding fusion set to `concat`, matching S2GNN composition:
  - base node projection: `188`
  - Laplacian PE projection: `150 -> 8`
  - RWSE projection: `20 -> 28` with `BatchNorm` on raw RW stats
- spectral basis count `60`
- cutoff `0.7`
- eigenspace cache `k_eig=150`
- RW encoding length `20`
- training: batch size `200`, lr `0.00225`, epochs `250`, checkpoint interval `100`
- optimizer/scheduler mapping: `AdamW`, `weight_decay=0.0035`,
  `cosine_with_warmup`, `num_warmup_epochs=5`, `clip_grad_norm=True`

Remaining differences from original GraphGym/S2GNN pipeline:

- GraphGym-specific split/runner internals
- exact S2GNN spatial/spectral block ordering
