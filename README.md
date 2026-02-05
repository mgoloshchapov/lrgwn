# Long-Range Graph Wavelet Networks (LR-GWN) – Implementation Notes

This repository implements the **Long-Range Graph Wavelet Networks** architecture described in *Guerranti et al., 2025*. The model follows the paper’s hybrid wavelet parametrization, combining a **spatial polynomial filter** with a **spectral low-rank correction**. Below is a concise, equation-first description of the implemented model and how it maps to the code.

## Graph Preliminaries
We use the symmetrically normalized Laplacian:

$$
L = I - D^{-1/2} A D^{-1/2}, \quad L = U \Lambda U^\top,
$$

where $U \in \mathbb{R}^{n \times n}$ is orthonormal and $\Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_n)$ with $0 = \lambda_1 \le \dots \le \lambda_n \le 2$. We use a **truncated spectrum** $(U_k, \Lambda_k)$ with $k \ll n$ (precomputed by `CachedSpectralTransform`).

Wavelet operators follow the spectral graph wavelet transform (SGWT):

$$
\Psi_s = U \, \hat\psi(s\Lambda) \, U^\top, \qquad \Phi = U \, \hat\varphi(\Lambda) \, U^\top.
$$

## Hybrid Filter Parametrization (Core LR‑GWN Idea)
At layer $l$, both the scaling filter $\hat\varphi^{(l)}$ and wavelet filter $\hat\psi^{(l)}$ are **sums of a polynomial (spatial) term and a spectral term**:

$$
\hat\psi^{(l)}(\Lambda) = P_{\omega_\psi^{(l)}}(\Lambda) + S_{\theta_\psi^{(l)}}(\Lambda),
$$
$$
\hat\varphi^{(l)}(\Lambda) = P_{\omega_\varphi^{(l)}}(\Lambda) + S_{\theta_\varphi^{(l)}}(\Lambda).
$$

The corresponding filtering operator applied to a signal $x$ is

$$
\psi^{(l)}(U,\Lambda,L)x = P_{\omega_\psi^{(l)}}(L)x + U\, S_{\theta_\psi^{(l)}}(\Lambda)\, U^\top x.
$$

This is implemented in `lrgwn/layer.py` as a **Chebyshev polynomial** (spatial) plus a **spectral low-rank correction** using the truncated eigenvectors.

## Spatial Part (Polynomial / Chebyshev)
The spatial component uses a Chebyshev polynomial of order $\rho$ (implemented via `ChebConv`):

$$
P_{\omega}(\Lambda) = \sum_{i=0}^{\rho} \omega_i \, T_i(\tilde\Lambda), \quad \tilde\Lambda = \frac{2}{\lambda_{\max}}\Lambda - I,
$$

with recurrence:

$$
T_0(x) = 1, \quad T_1(x) = x, \quad T_i(x) = 2xT_{i-1}(x) - T_{i-2}(x).
$$

In the vertex domain this becomes $P_{\omega}(L)x$. The code uses `ChebConv(..., normalization="sym")`, which internally uses $\lambda_{\max}$ from `LaplacianLambdaMax`.

## Spectral Part (Low‑Rank Correction)
The spectral component is defined on eigenvalues and parameterized by Gaussian smearing:

$$
S_{\theta^{(l)}}(\lambda) = \mathrm{GaussianSmearing}(\lambda) \, W_{\theta^{(l)}},
$$

where the smearing maps each eigenvalue to $z$ Gaussian radial basis functions. In code, for centers $\mu_j$ spaced in $[0, \lambda_{\text{cut}}]$ and width $\sigma$:

$$
\mathrm{GaussianSmearing}(\lambda)_j = \exp\Big( -\frac{(\lambda-\mu_j)^2}{2\sigma^2} \Big).
$$

We then apply the spectral correction through the truncated eigenspace:

$$
U_k\, S_{\theta}(\Lambda_k)\, U_k^\top x.
$$

The current implementation sets $\lambda_{\text{cut}}=2.0$ (consistent with the normalized Laplacian spectrum) and uses a learnable linear map $W_{\theta}$.

## Shared vs. Independent Wavelet Filters
The paper supports **shared** or **independent** wavelet parametrizations, both implemented:

Shared (single mother wavelet with scales $s_j$):

$$
\psi(U,\Lambda,L; s_j) = U\,S_{\theta_\psi}(s_j\Lambda)\,U^\top + P_{\omega_\psi}(s_j L).
$$

Independent (each wavelet has its own parameters):

$$
\psi_j(U,\Lambda,L) = U\,S_{\theta_{\psi_j}}(\Lambda)\,U^\top + P_{\omega_{\psi_j}}(L).
$$

## Scale Parametrization (Shared Filters)
For shared filters, the paper defines learnable scale bounds and log‑space interpolation. We mirror this behavior by learning $s_{\min}$ and $s_{\max}$ and interpolating in log‑space:

$$
\log s_i = \log s_{\max} + \frac{i}{N-1}(\log s_{\min} - \log s_{\max}), \quad i=0,\dots,N-1.
$$

In code, $s_{\min}$ and $s_{\max}$ are learned with a softplus‑style constraint to ensure positivity.

## Admissibility (Optional)
Wavelet admissibility requires $\hat\psi(0)=0$. The paper enforces it by subtracting the zero‑frequency response:

$$
\tilde P_{\omega}(\lambda)=P_{\omega}(\lambda)-P_{\omega}(0), \quad
\tilde S_{\theta}(\lambda)=S_{\theta}(\lambda)-S_{\theta}(0).
$$

Then $\hat\psi(\Lambda)=\tilde P_{\omega}(\Lambda)+\tilde S_{\theta}(\Lambda)$ ensures $\hat\psi(0)=0$. The implementation supports this via the `admissible` flag.

## Layer Output and Aggregation
Each layer first projects features $x$ and then computes the scaling filter plus $J$ wavelet filters. We aggregate them either by sum or concatenation:

Sum:

$$
H^{(l)} = \sum_{j=0}^{J} \sigma\big(\psi_j^{(l)}(U,\Lambda,L)\,x\big)
$$

Concat:

$$
H^{(l)} = W_{\text{merge}}\,[\sigma(\psi_0^{(l)}x) \| \cdots \| \sigma(\psi_J^{(l)}x)],
$$

where $\psi_0$ denotes the scaling function and $\sigma$ is ReLU. Dropout is applied after aggregation.

## Full Model Architecture
The model used in `lrgwn/model.py` is:

$$
X^{(0)} = W_{\text{in}} X,
$$
$$
X^{(l+1)} = \mathrm{LRGWNLayer}(X^{(l)}, U_k, \Lambda_k, L), \quad l=0,\dots,L-1,
$$
$$
\hat y = \mathrm{MLP}(\mathrm{GlobalMeanPool}(X^{(L)})).
$$

This matches the LR‑GWN structure for graph classification on Peptides‑func.

---

If you want the README to include additional equations (e.g., windowing functions for spectral regularization or residual wavelet variants), tell me which sections of the paper to incorporate and I will update it accordingly.
