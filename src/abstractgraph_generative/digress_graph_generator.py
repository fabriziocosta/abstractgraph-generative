"""
Single-file discrete graph generator based on DiGress.

Matched from DiGress:
- `src/diffusion/noise_schedule.py:PredefinedNoiseScheduleDiscrete`
- `src/diffusion/noise_schedule.py:MarginalUniformTransition`
- `src/diffusion/noise_schedule.py:DiscreteUniformTransition`
- `src/diffusion/diffusion_utils.py:cosine_beta_schedule_discrete`
- `src/diffusion/diffusion_utils.py:sample_discrete_features`
- `src/diffusion/diffusion_utils.py:compute_posterior_distribution`
- `src/diffusion/diffusion_utils.py:compute_batched_over0_posterior_distribution`
- `src/diffusion/diffusion_utils.py:mask_distributions`
- `src/diffusion/diffusion_utils.py:posterior_distributions`
- `src/diffusion/diffusion_utils.py:sample_discrete_feature_noise`
- `src/models/layers.py:Xtoy`
- `src/models/layers.py:Etoy`
- `src/models/layers.py:masked_softmax`
- `src/models/transformer_model.py:NodeEdgeBlock`
- `src/models/transformer_model.py:XEyTransformerLayer`
- `src/models/transformer_model.py:GraphTransformer`
- `src/metrics/train_metrics.py:TrainLossDiscrete`
- `src/diffusion_model_discrete.py:apply_noise`
- `src/diffusion_model_discrete.py:sample_p_zs_given_zt`
- `src/diffusion_model_discrete.py:compute_Lt`
- `src/diffusion_model_discrete.py:reconstruction_logp`
- `src/diffusion_model_discrete.py:compute_val_loss`

Exact loss implemented:
- Training uses the official DiGress discrete surrogate from
  `TrainLossDiscrete.forward`: cross-entropy on node classes plus
  `lambda_train[0] *` cross-entropy on edge classes. DiGress sets
  `lambda_train = [5, 0]` by default, so the edge term is weighted by 5 and
  the global `y` term is omitted here because this wrapper has no graph-level
  targets.
- Forward noising, reverse posterior construction, and optional variational NLL
  helpers follow the official discrete DiGress formulas above.

Assumptions and limitations:
- Undirected simple `nx.Graph` only. Directed graphs, `MultiGraph`, and
  `MultiDiGraph` raise `ValueError`.
- Edge type `0` is reserved for `NO_EDGE`, matching DiGress.
- Nodes do not use PAD or UNK classes. This matches DiGress: padding is handled
  by masks and padded rows are all zeros, not a categorical token.
- Diagonal edge entries are kept as all-zero masked rows, matching DiGress'
  dense representation.
- Pure PyTorch implementation; no PyG dependency.
- Practical hard cap: graphs with more than 50 nodes raise `ValueError`.

License notice:
- This file includes adapted logic from the official DiGress repository
  (MIT License, Copyright (c) 2012-2022 Clement Vignac, Igor Krawczuk,
  Antoine Siraudin). The original copyright and permission notice apply.
"""

from __future__ import annotations

import math
import os
import random
import copy
import hashlib
import tempfile
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def _set_seed(seed: int) -> None:
    """Set deterministic seeds as far as PyTorch allows.

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")


@dataclass
class _PlaceHolder:
    """Small DiGress-style container."""

    X: torch.Tensor
    E: torch.Tensor
    y: torch.Tensor

    def type_as(self, x: torch.Tensor) -> "_PlaceHolder":
        """Move tensors to the same dtype/device as `x`.

        Args:
            x: Reference tensor.

        Returns:
            Updated placeholder.
        """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask: torch.Tensor, collapse: bool = False) -> "_PlaceHolder":
        """Apply DiGress masking rules.

        Args:
            node_mask: Boolean node mask with shape `[bs, n]`.
            collapse: If true, collapse one-hot tensors to integer ids.

        Returns:
            Updated placeholder.
        """
        x_mask = node_mask.unsqueeze(-1)
        e_mask1 = x_mask.unsqueeze(2)
        e_mask2 = x_mask.unsqueeze(1)
        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)
            self.X[node_mask == 0] = -1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            if self.E.numel():
                assert torch.allclose(self.E, self.E.transpose(1, 2))
        return self


class _DistributionNodes:
    """Empirical node-count distribution."""

    def __init__(self, histogram: dict[int, int] | torch.Tensor):
        if isinstance(histogram, dict):
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1, dtype=torch.float32)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = float(count)
        else:
            prob = histogram.float()
        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(self.prob)

    def sample_n(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """Sample node counts.

        Args:
            n_samples: Number of samples.
            device: Output device.

        Returns:
            Integer tensor of node counts.
        """
        return self.m.sample((n_samples,)).to(device)

    def log_prob(self, batch_n_nodes: torch.Tensor) -> torch.Tensor:
        """Log-probability of node counts.

        Args:
            batch_n_nodes: Tensor of shape `[bs]`.

        Returns:
            Log-probabilities.
        """
        p = self.prob.to(batch_n_nodes.device)
        return torch.log(p[batch_n_nodes] + 1e-30)


def _sum_except_batch(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.size(0), -1).sum(dim=-1)


def _assert_correctly_masked(variable: torch.Tensor, node_mask: torch.Tensor) -> None:
    if variable.numel() == 0:
        return
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4


def _cosine_beta_schedule_discrete(timesteps: int, s: float = 0.008) -> np.ndarray:
    """Copied from DiGress discrete cosine schedule.

    Args:
        timesteps: Number of diffusion steps.
        s: Cosine schedule offset.

    Returns:
        Beta schedule as a NumPy array.
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    return betas.squeeze()


class _PredefinedNoiseScheduleDiscrete(nn.Module):
    """DiGress discrete noise schedule."""

    def __init__(self, noise_schedule: str, timesteps: int):
        super().__init__()
        self.timesteps = timesteps
        if noise_schedule != "cosine":
            raise RuntimeError(
                "Only the DiGress discrete cosine schedule is implemented. "
                "Missing source match for alternative schedule."
            )
        betas = _cosine_beta_schedule_discrete(timesteps)
        self.register_buffer("betas", torch.from_numpy(betas).float())
        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)
        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)

    def forward(
        self,
        t_normalized: torch.Tensor | None = None,
        t_int: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return beta_t.

        Args:
            t_normalized: Normalized time in `[0, 1]`.
            t_int: Integer step in `[0, T]`.

        Returns:
            Beta values.
        """
        if (t_normalized is None) == (t_int is None):
            raise ValueError("Exactly one of t_normalized or t_int must be set.")
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas[t_int.long()]

    def get_alpha_bar(
        self,
        t_normalized: torch.Tensor | None = None,
        t_int: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return alpha_bar_t.

        Args:
            t_normalized: Normalized time in `[0, 1]`.
            t_int: Integer step in `[0, T]`.

        Returns:
            Alpha-bar values.
        """
        if (t_normalized is None) == (t_int is None):
            raise ValueError("Exactly one of t_normalized or t_int must be set.")
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.alphas_bar.to(t_int.device)[t_int.long()]


class _DiscreteUniformTransition:
    """DiGress uniform categorical transition."""

    def __init__(self, x_classes: int, e_classes: int, y_classes: int):
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes
        self.u_x = torch.ones(1, self.X_classes, self.X_classes) / max(self.X_classes, 1)
        self.u_e = torch.ones(1, self.E_classes, self.E_classes) / max(self.E_classes, 1)
        self.u_y = torch.ones(1, self.y_classes, self.y_classes) / max(self.y_classes, 1)

    def get_Qt(self, beta_t: torch.Tensor, device: torch.device) -> _PlaceHolder:
        """Return one-step transition matrices.

        Args:
            beta_t: Noise level with shape `[bs, 1]` or `[bs]`.
            device: Target device.

        Returns:
            Transition matrices.
        """
        beta_t = beta_t.unsqueeze(1).to(device)
        u_x = self.u_x.to(device)
        u_e = self.u_e.to(device)
        u_y = self.u_y.to(device)
        q_x = beta_t * u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)
        q_e = beta_t * u_e + (1 - beta_t) * torch.eye(self.E_classes, device=device).unsqueeze(0)
        q_y = beta_t * u_y + (1 - beta_t) * torch.eye(self.y_classes, device=device).unsqueeze(0)
        return _PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t: torch.Tensor, device: torch.device) -> _PlaceHolder:
        """Return cumulative transition matrices.

        Args:
            alpha_bar_t: Alpha-bar values with shape `[bs, 1]` or `[bs]`.
            device: Target device.

        Returns:
            Transition matrices.
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1).to(device)
        u_x = self.u_x.to(device)
        u_e = self.u_e.to(device)
        u_y = self.u_y.to(device)
        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * u_x
        q_e = alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * u_e
        q_y = alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * u_y
        return _PlaceHolder(X=q_x, E=q_e, y=q_y)


class _MarginalUniformTransition:
    """DiGress marginal categorical transition."""

    def __init__(self, x_marginals: torch.Tensor, e_marginals: torch.Tensor, y_classes: int):
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.y_classes = y_classes
        self.u_x = x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)
        self.u_e = e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)
        self.u_y = torch.ones(1, self.y_classes, self.y_classes) / max(self.y_classes, 1)

    def get_Qt(self, beta_t: torch.Tensor, device: torch.device) -> _PlaceHolder:
        """Return one-step transition matrices.

        Args:
            beta_t: Noise level.
            device: Target device.

        Returns:
            Transition matrices.
        """
        beta_t = beta_t.unsqueeze(1).to(device)
        u_x = self.u_x.to(device)
        u_e = self.u_e.to(device)
        u_y = self.u_y.to(device)
        q_x = beta_t * u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)
        q_e = beta_t * u_e + (1 - beta_t) * torch.eye(self.E_classes, device=device).unsqueeze(0)
        q_y = beta_t * u_y + (1 - beta_t) * torch.eye(self.y_classes, device=device).unsqueeze(0)
        return _PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t: torch.Tensor, device: torch.device) -> _PlaceHolder:
        """Return cumulative transition matrices.

        Args:
            alpha_bar_t: Alpha-bar values.
            device: Target device.

        Returns:
            Transition matrices.
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1).to(device)
        u_x = self.u_x.to(device)
        u_e = self.u_e.to(device)
        u_y = self.u_y.to(device)
        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * u_x
        q_e = alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * u_e
        q_y = alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * u_y
        return _PlaceHolder(X=q_x, E=q_e, y=q_y)


def _sample_discrete_features(
    probX: torch.Tensor,
    probE: torch.Tensor,
    node_mask: torch.Tensor,
    generator: torch.Generator | None,
) -> _PlaceHolder:
    """Sample categorical node and edge features exactly as in DiGress.

    Args:
        probX: Node probabilities `[bs, n, dx]`.
        probE: Edge probabilities `[bs, n, n, de]`.
        node_mask: Node mask `[bs, n]`.
        generator: Torch generator.

    Returns:
        Integer node and edge ids.
    """
    bs, n, _ = probX.shape
    probX = probX.clone()
    probE = probE.clone()
    probX[~node_mask] = 1 / probX.shape[-1]
    probX = probX.reshape(bs * n, -1)
    X_t = torch.multinomial(probX, 1, generator=generator).reshape(bs, n)

    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n, device=probE.device, dtype=torch.bool).unsqueeze(0).expand(bs, -1, -1)
    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask] = 1 / probE.shape[-1]
    probE = probE.reshape(bs * n * n, -1)
    E_t = torch.multinomial(probE, 1, generator=generator).reshape(bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = E_t + E_t.transpose(1, 2)
    return _PlaceHolder(X=X_t, E=E_t, y=torch.zeros(bs, 0, device=X_t.device, dtype=X_t.dtype))


def _compute_posterior_distribution(
    M: torch.Tensor,
    M_t: torch.Tensor,
    Qt_M: torch.Tensor,
    Qsb_M: torch.Tensor,
    Qtb_M: torch.Tensor,
) -> torch.Tensor:
    M = M.flatten(start_dim=1, end_dim=-2).to(torch.float32)
    M_t = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)
    Qt_M_T = Qt_M.transpose(-2, -1)
    left_term = M_t @ Qt_M_T
    right_term = M @ Qsb_M
    product = left_term * right_term
    denom = M @ Qtb_M
    denom = (denom * M_t).sum(dim=-1)
    return product / denom.unsqueeze(-1)


def _compute_batched_over0_posterior_distribution(
    X_t: torch.Tensor,
    Qt: torch.Tensor,
    Qsb: torch.Tensor,
    Qtb: torch.Tensor,
) -> torch.Tensor:
    X_t = X_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)
    Qt_T = Qt.transpose(-1, -2)
    left_term = X_t @ Qt_T
    left_term = left_term.unsqueeze(dim=2)
    right_term = Qsb.unsqueeze(1)
    numerator = left_term * right_term
    X_t_transposed = X_t.transpose(-1, -2)
    prod = Qtb @ X_t_transposed
    prod = prod.transpose(-1, -2)
    denominator = prod.unsqueeze(-1)
    denominator[denominator == 0] = 1e-6
    return numerator / denominator


def _mask_distributions(
    true_X: torch.Tensor,
    true_E: torch.Tensor,
    pred_X: torch.Tensor,
    pred_E: torch.Tensor,
    node_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    row_X = torch.zeros(true_X.size(-1), dtype=torch.float, device=true_X.device)
    row_X[0] = 1.0
    row_E = torch.zeros(true_E.size(-1), dtype=torch.float, device=true_E.device)
    row_E[0] = 1.0
    diag_mask = ~torch.eye(node_mask.size(1), device=node_mask.device, dtype=torch.bool).unsqueeze(0)
    true_X = true_X.clone()
    true_E = true_E.clone()
    pred_X = pred_X.clone()
    pred_E = pred_E.clone()
    true_X[~node_mask] = row_X
    pred_X[~node_mask] = row_X
    valid_edges = node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask
    true_E[~valid_edges, :] = row_E
    pred_E[~valid_edges, :] = row_E
    true_X = true_X + 1e-7
    pred_X = pred_X + 1e-7
    true_E = true_E + 1e-7
    pred_E = pred_E + 1e-7
    true_X = true_X / torch.sum(true_X, dim=-1, keepdim=True)
    pred_X = pred_X / torch.sum(pred_X, dim=-1, keepdim=True)
    true_E = true_E / torch.sum(true_E, dim=-1, keepdim=True)
    pred_E = pred_E / torch.sum(pred_E, dim=-1, keepdim=True)
    return true_X, true_E, pred_X, pred_E


def _posterior_distributions(
    X: torch.Tensor,
    E: torch.Tensor,
    y: torch.Tensor,
    X_t: torch.Tensor,
    E_t: torch.Tensor,
    y_t: torch.Tensor,
    Qt: _PlaceHolder,
    Qsb: _PlaceHolder,
    Qtb: _PlaceHolder,
) -> _PlaceHolder:
    prob_X = _compute_posterior_distribution(M=X, M_t=X_t, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X)
    prob_E = _compute_posterior_distribution(M=E, M_t=E_t, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E)
    return _PlaceHolder(X=prob_X, E=prob_E, y=y_t if y.numel() == 0 else y)


def _sample_discrete_feature_noise(
    limit_dist: _PlaceHolder,
    node_mask: torch.Tensor,
    generator: torch.Generator | None,
) -> _PlaceHolder:
    bs, n_max = node_mask.shape
    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1)
    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
    y_limit = limit_dist.y[None, :].expand(bs, -1)
    U_X = torch.multinomial(x_limit.flatten(end_dim=-2), 1, generator=generator).reshape(bs, n_max)
    U_E = torch.multinomial(e_limit.flatten(end_dim=-2), 1, generator=generator).reshape(bs, n_max, n_max)
    U_y = torch.empty((bs, y_limit.shape[-1]), device=node_mask.device)
    U_X = U_X.type_as(node_mask.long())
    U_E = U_E.type_as(node_mask.long())
    U_y = U_y.type_as(node_mask.long())
    U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
    U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()
    upper_triangular_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1, device=U_E.device)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1
    U_E = U_E * upper_triangular_mask
    U_E = U_E + U_E.transpose(1, 2)
    return _PlaceHolder(X=U_X, E=U_E, y=U_y).mask(node_mask)


class _Xtoy(nn.Module):
    def __init__(self, dx: int, dy: int):
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.hstack((m, mi, ma, std))
        return self.lin(z)


class _Etoy(nn.Module):
    def __init__(self, d: int, dy: int):
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        return self.lin(z)


def _masked_softmax(x: torch.Tensor, mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)


class _NodeEdgeBlock(nn.Module):
    """DiGress node-edge self-attention block."""

    def __init__(self, dx: int, de: int, dy: int, n_head: int):
        super().__init__()
        if dx % n_head != 0:
            raise ValueError(f"dx must be divisible by n_head, got dx={dx}, n_head={n_head}")
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = dx // n_head
        self.n_head = n_head
        self.q = nn.Linear(dx, dx)
        self.k = nn.Linear(dx, dx)
        self.v = nn.Linear(dx, dx)
        self.e_add = nn.Linear(de, dx)
        self.e_mul = nn.Linear(de, dx)
        self.y_e_mul = nn.Linear(dy, dx)
        self.y_e_add = nn.Linear(dy, dx)
        self.y_x_mul = nn.Linear(dy, dx)
        self.y_x_add = nn.Linear(dy, dx)
        self.y_y = nn.Linear(dy, dy)
        self.x_y = _Xtoy(dx, dy)
        self.e_y = _Etoy(de, dy)
        self.x_out = nn.Linear(dx, dx)
        self.e_out = nn.Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        y: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)
        e_mask1 = x_mask.unsqueeze(2)
        e_mask2 = x_mask.unsqueeze(1)
        Q = self.q(X) * x_mask
        K = self.k(X) * x_mask
        _assert_correctly_masked(Q, x_mask)
        Q = Q.reshape(bs, n, self.n_head, self.df).unsqueeze(2)
        K = K.reshape(bs, n, self.n_head, self.df).unsqueeze(1)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        _assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))
        E1 = self.e_mul(E) * e_mask1 * e_mask2
        E1 = E1.reshape(bs, n, n, self.n_head, self.df)
        E2 = self.e_add(E) * e_mask1 * e_mask2
        E2 = E2.reshape(bs, n, n, self.n_head, self.df)
        Y = Y * (E1 + 1) + E2
        newE = Y.flatten(start_dim=3)
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE
        newE = self.e_out(newE) * e_mask1 * e_mask2
        _assert_correctly_masked(newE, e_mask1 * e_mask2)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)
        attn = _masked_softmax(Y, softmax_mask, dim=2)
        V = self.v(X) * x_mask
        V = V.reshape(bs, n, self.n_head, self.df).unsqueeze(1)
        weighted_V = (attn * V).sum(dim=2).flatten(start_dim=2)
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V
        newX = self.x_out(newX) * x_mask
        _assert_correctly_masked(newX, x_mask)
        new_y = self.y_y(y) + self.x_y(X) + self.e_y(E)
        new_y = self.y_out(new_y)
        return newX, newE, new_y


class _XEyTransformerLayer(nn.Module):
    """DiGress transformer layer on nodes, edges, and globals."""

    def __init__(
        self,
        dx: int,
        de: int,
        dy: int,
        n_head: int,
        dim_ffX: int,
        dim_ffE: int,
        dim_ffy: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = _NodeEdgeBlock(dx, de, dy, n_head)
        self.linX1 = nn.Linear(dx, dim_ffX)
        self.linX2 = nn.Linear(dim_ffX, dx)
        self.normX1 = nn.LayerNorm(dx)
        self.normX2 = nn.LayerNorm(dx)
        self.dropoutX1 = nn.Dropout(dropout)
        self.dropoutX2 = nn.Dropout(dropout)
        self.dropoutX3 = nn.Dropout(dropout)
        self.linE1 = nn.Linear(de, dim_ffE)
        self.linE2 = nn.Linear(dim_ffE, de)
        self.normE1 = nn.LayerNorm(de)
        self.normE2 = nn.LayerNorm(de)
        self.dropoutE1 = nn.Dropout(dropout)
        self.dropoutE2 = nn.Dropout(dropout)
        self.dropoutE3 = nn.Dropout(dropout)
        self.lin_y1 = nn.Linear(dy, dim_ffy)
        self.lin_y2 = nn.Linear(dim_ffy, dy)
        self.norm_y1 = nn.LayerNorm(dy)
        self.norm_y2 = nn.LayerNorm(dy)
        self.dropout_y1 = nn.Dropout(dropout)
        self.dropout_y2 = nn.Dropout(dropout)
        self.dropout_y3 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        y: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)
        X = self.normX1(X + self.dropoutX1(newX))
        E = self.normE1(E + self.dropoutE1(newE))
        y = self.norm_y1(y + self.dropout_y1(new_y))
        X = self.normX2(X + self.dropoutX3(self.linX2(self.dropoutX2(self.activation(self.linX1(X))))))
        E = self.normE2(E + self.dropoutE3(self.linE2(self.dropoutE2(self.activation(self.linE1(E))))))
        y = self.norm_y2(y + self.dropout_y3(self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))))
        return X, E, y


class _ZeroOutput(nn.Module):
    """Helper module for zero-sized output heads."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.shape[0], 0)


class _GraphTransformer(nn.Module):
    """DiGress graph transformer with the same block structure."""

    def __init__(
        self,
        n_layers: int,
        input_dims: dict[str, int],
        hidden_mlp_dims: dict[str, int],
        hidden_dims: dict[str, int],
        output_dims: dict[str, int],
    ):
        super().__init__()
        self.out_dim_X = output_dims["X"]
        self.out_dim_E = output_dims["E"]
        self.out_dim_y = output_dims["y"]
        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims["X"], hidden_mlp_dims["X"]),
            nn.ReLU(),
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]),
            nn.ReLU(),
        )
        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims["E"], hidden_mlp_dims["E"]),
            nn.ReLU(),
            nn.Linear(hidden_mlp_dims["E"], hidden_dims["de"]),
            nn.ReLU(),
        )
        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims["y"], hidden_mlp_dims["y"]),
            nn.ReLU(),
            nn.Linear(hidden_mlp_dims["y"], hidden_dims["dy"]),
            nn.ReLU(),
        )
        self.tf_layers = nn.ModuleList(
            [
                _XEyTransformerLayer(
                    dx=hidden_dims["dx"],
                    de=hidden_dims["de"],
                    dy=hidden_dims["dy"],
                    n_head=hidden_dims["n_head"],
                    dim_ffX=hidden_dims["dim_ffX"],
                    dim_ffE=hidden_dims["dim_ffE"],
                    dim_ffy=hidden_dims["dim_ffy"],
                )
                for _ in range(n_layers)
            ]
        )
        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]),
            nn.ReLU(),
            nn.Linear(hidden_mlp_dims["X"], output_dims["X"]),
        )
        self.mlp_out_E = nn.Sequential(
            nn.Linear(hidden_dims["de"], hidden_mlp_dims["E"]),
            nn.ReLU(),
            nn.Linear(hidden_mlp_dims["E"], output_dims["E"]),
        )
        self.mlp_out_y = (
            _ZeroOutput()
            if output_dims["y"] == 0
            else nn.Sequential(
                nn.Linear(hidden_dims["dy"], hidden_mlp_dims["y"]),
                nn.ReLU(),
                nn.Linear(hidden_mlp_dims["y"], output_dims["y"]),
            )
        )

    def forward(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        y: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> _PlaceHolder:
        bs, n = X.shape[:2]
        diag_mask = ~torch.eye(n, device=E.device, dtype=torch.bool).unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)
        X_to_out = X[..., : self.out_dim_X]
        E_to_out = E[..., : self.out_dim_E]
        y_to_out = y[..., : self.out_dim_y] if self.out_dim_y > 0 else y.new_zeros(bs, 0)
        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        after_in = _PlaceHolder(
            X=self.mlp_in_X(X),
            E=new_E,
            y=self.mlp_in_y(y),
        ).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y
        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)
        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)
        X = X + X_to_out
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out
        E = 0.5 * (E + E.transpose(1, 2))
        return _PlaceHolder(X=X, E=E, y=y).mask(node_mask)


class _GraphDataset(Dataset):
    """Tensorized graph dataset."""

    def __init__(self, samples: list[dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


class _EmptyFeatures:
    """No-op extra feature provider matching DiGress API shape."""

    def __call__(self, noisy_data: dict[str, torch.Tensor]) -> _PlaceHolder:
        X_t = noisy_data["X_t"]
        E_t = noisy_data["E_t"]
        bs, n = X_t.shape[:2]
        return _PlaceHolder(
            X=X_t.new_zeros(bs, n, 0),
            E=E_t.new_zeros(bs, n, n, 0),
            y=X_t.new_zeros(bs, 0),
        )


class DiGressGraphGenerator:
    """Single-file DiGress-style discrete graph generator.

    The implementation matches DiGress' discrete transition kernels and reverse
    posterior parameterization, while keeping the wrapper limited to plain
    NetworkX graphs with discrete node and edge labels.

    Key constructor hyperparameters:
    - `diffusion_steps`: more steps usually improve fidelity but increase both
      training and sampling time linearly.
    - `hidden_mlp_dims` and `hidden_dims`: larger widths increase model capacity
      and memory use; smaller widths train faster but may underfit.
    - `early_stopping_patience`: larger patience is more tolerant of noisy
      validation curves; smaller patience stops sooner.
    - `real_loss_log_every` and `real_loss_subset_size`: control how often to
      evaluate the expensive variational-style loss and on how many training
      graphs.
    """

    _MAX_NODES = 50
    _NO_EDGE = "__NO_EDGE__"
    _REQUIRED_HIDDEN_MLP_DIM_KEYS = ("X", "E", "y")
    _REQUIRED_HIDDEN_DIM_KEYS = ("dx", "de", "dy", "n_head", "dim_ffX", "dim_ffE", "dim_ffy")

    def __init__(
        self,
        *,
        diffusion_steps: int = 100,
        n_layers: int = 3,
        hidden_mlp_dims: dict[str, int] | None = None,
        hidden_dims: dict[str, int] | None = None,
        early_stopping_patience: int = 5,
        real_loss_log_every: int = 5,
        real_loss_subset_size: int = 256,
    ) -> None:
        """Initialize the generator architecture and training controls.

        Args:
            diffusion_steps: Number of discrete diffusion steps. Higher values
                generally improve expressiveness but slow down training and
                sampling.
            n_layers: Number of DiGress transformer layers. More layers can fit
                more complex graph structure but raise compute cost.
            hidden_mlp_dims: Widths of the input/output MLPs for node, edge,
                and global channels.
            hidden_dims: Core transformer widths and feed-forward sizes. These
                dominate memory usage and model capacity.
            early_stopping_patience: Number of validation epochs without
                improvement before stopping training.
            real_loss_log_every: Evaluate and print the expensive DiGress
                variational-style train loss every k epochs.
            real_loss_subset_size: Size of the fixed training subset used when
                computing the logged real loss.
        """
        self._is_fitted = False
        self._device = torch.device("cpu")
        self._train_seed = 0
        self._sample_seed_offset = 100_000
        self._node_label_key = "label"
        self._edge_label_key = "label"
        self._node_label_to_id: dict[Any, int] = {}
        self._id_to_node_label: list[Any] = []
        self._edge_label_to_id: dict[Any, int] = {self._NO_EDGE: 0}
        self._id_to_edge_label: list[Any] = [self._NO_EDGE]
        self._node_dist: _DistributionNodes | None = None
        self._transition_model: _MarginalUniformTransition | _DiscreteUniformTransition | None = None
        self._limit_dist: _PlaceHolder | None = None
        self._noise_schedule: _PredefinedNoiseScheduleDiscrete | None = None
        self._model: _GraphTransformer | None = None
        self._generator: torch.Generator | None = None
        if diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be positive.")
        if n_layers <= 0:
            raise ValueError("n_layers must be positive.")
        if early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative.")
        if real_loss_log_every <= 0:
            raise ValueError("real_loss_log_every must be positive.")
        if real_loss_subset_size <= 0:
            raise ValueError("real_loss_subset_size must be positive.")
        self._T = diffusion_steps
        self._lambda_train = (5.0, 0.0)
        hidden_mlp_dims = hidden_mlp_dims or {"X": 128, "E": 64, "y": 64}
        hidden_dims = hidden_dims or {
            "dx": 128,
            "de": 64,
            "dy": 64,
            "n_head": 8,
            "dim_ffX": 128,
            "dim_ffE": 64,
            "dim_ffy": 128,
        }
        self._validate_dim_dict("hidden_mlp_dims", hidden_mlp_dims, self._REQUIRED_HIDDEN_MLP_DIM_KEYS)
        self._validate_dim_dict("hidden_dims", hidden_dims, self._REQUIRED_HIDDEN_DIM_KEYS)
        self._hidden_mlp_dims = dict(hidden_mlp_dims)
        self._hidden_dims = dict(hidden_dims)
        if self._hidden_dims["dx"] % self._hidden_dims["n_head"] != 0:
            raise ValueError("hidden_dims['dx'] must be divisible by hidden_dims['n_head'].")
        self._n_layers = n_layers
        self._extra_features = _EmptyFeatures()
        self._domain_features = _EmptyFeatures()
        self._dataset: list[dict[str, Any]] = []
        self._class_states: dict[Any, dict[str, Any]] = {}
        self._active_target: Any | None = None
        self._verbose = False
        self._loss_history: dict[Any, list[float]] = {}
        self._real_loss_history: dict[Any, list[float]] = {}
        self._val_loss_history: dict[Any, list[float]] = {}
        self._early_stopping_patience = early_stopping_patience
        self._real_loss_log_every = real_loss_log_every
        self._real_loss_subset_size = real_loss_subset_size
        self._checkpoint_dir: str | None = None

    @classmethod
    def _validate_dim_dict(
        cls,
        name: str,
        values: dict[str, int],
        required_keys: tuple[str, ...],
    ) -> None:
        """Validate required keys and positive integer values for dim dicts.

        Args:
            name: Human-readable parameter name.
            values: Dictionary to validate.
            required_keys: Required keys for the dictionary.
        """
        missing = [key for key in required_keys if key not in values]
        extra = [key for key in values if key not in required_keys]
        if missing or extra:
            problems = []
            if missing:
                problems.append(f"missing keys: {missing}")
            if extra:
                problems.append(f"unexpected keys: {extra}")
            raise ValueError(f"{name} is invalid; " + "; ".join(problems))
        for key in required_keys:
            value = values[key]
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name}[{key!r}] must be a positive integer, got {value!r}.")

    @staticmethod
    def _canonical_label_order(values: set[Any]) -> list[Any]:
        return sorted(values, key=lambda v: (type(v).__name__, repr(v)))

    @staticmethod
    def _validate_graph(graph: nx.Graph) -> None:
        if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
            raise ValueError("Only undirected simple nx.Graph inputs are supported; MultiGraph inputs are invalid.")
        if graph.is_directed():
            raise ValueError("Directed graphs are not supported; pass an undirected nx.Graph.")

    def _build_vocabularies(self, graphs: list[nx.Graph]) -> None:
        node_labels: set[Any] = set()
        edge_labels: set[Any] = set()
        node_hist: dict[int, int] = {}
        for graph in graphs:
            n = graph.number_of_nodes()
            if n <= 0:
                raise ValueError("Empty graphs are not supported.")
            if n > self._MAX_NODES:
                raise ValueError(f"Graphs with more than {self._MAX_NODES} nodes are not supported.")
            node_hist[n] = node_hist.get(n, 0) + 1
            for _, attrs in graph.nodes(data=True):
                if self._node_label_key not in attrs:
                    raise ValueError(f"Missing node attribute '{self._node_label_key}'.")
                node_labels.add(attrs[self._node_label_key])
            for _, _, attrs in graph.edges(data=True):
                if self._edge_label_key not in attrs:
                    raise ValueError(f"Missing edge attribute '{self._edge_label_key}'.")
                edge_labels.add(attrs[self._edge_label_key])
        self._id_to_node_label = self._canonical_label_order(node_labels)
        self._node_label_to_id = {label: idx for idx, label in enumerate(self._id_to_node_label)}
        ordered_edges = self._canonical_label_order(edge_labels)
        self._id_to_edge_label = [self._NO_EDGE] + ordered_edges
        self._edge_label_to_id = {label: idx for idx, label in enumerate(self._id_to_edge_label)}
        self._node_dist = _DistributionNodes(node_hist)

    def _encode_graph(self, graph: nx.Graph) -> dict[str, Any]:
        self._validate_graph(graph)
        nodes = list(graph.nodes())
        n = len(nodes)
        idx_of = {node: idx for idx, node in enumerate(nodes)}
        X_ids = torch.empty(n, dtype=torch.long)
        for node, idx in idx_of.items():
            label = graph.nodes[node][self._node_label_key]
            X_ids[idx] = self._node_label_to_id[label]
        E_ids = torch.zeros((n, n), dtype=torch.long)
        for u, v, attrs in graph.edges(data=True):
            label = attrs[self._edge_label_key]
            edge_id = self._edge_label_to_id[label]
            iu = idx_of[u]
            iv = idx_of[v]
            E_ids[iu, iv] = edge_id
            E_ids[iv, iu] = edge_id
        X = F.one_hot(X_ids, num_classes=len(self._id_to_node_label)).float()
        E = F.one_hot(E_ids, num_classes=len(self._id_to_edge_label)).float()
        diag = torch.eye(n, dtype=torch.bool)
        E[diag] = 0.0
        return {"X": X, "E": E, "n": n}

    def _collate(self, batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_n = max(sample["n"] for sample in batch)
        bs = len(batch)
        xdim = len(self._id_to_node_label)
        edim = len(self._id_to_edge_label)
        X = torch.zeros(bs, max_n, xdim, dtype=torch.float32)
        E = torch.zeros(bs, max_n, max_n, edim, dtype=torch.float32)
        node_mask = torch.zeros(bs, max_n, dtype=torch.bool)
        for i, sample in enumerate(batch):
            n = sample["n"]
            X[i, :n] = sample["X"]
            E[i, :n, :n] = sample["E"]
            node_mask[i, :n] = True
        return X, E, node_mask

    def _compute_marginals(self) -> tuple[torch.Tensor, torch.Tensor]:
        node_counts = torch.zeros(len(self._id_to_node_label), dtype=torch.float32)
        edge_counts = torch.zeros(len(self._id_to_edge_label), dtype=torch.float32)
        for sample in self._dataset:
            X_ids = sample["X"].argmax(dim=-1)
            node_counts += torch.bincount(X_ids, minlength=node_counts.numel()).float()
            E_ids = sample["E"].argmax(dim=-1)
            n = sample["n"]
            valid = ~torch.eye(n, dtype=torch.bool)
            edge_counts += torch.bincount(E_ids[valid], minlength=edge_counts.numel()).float()
        node_counts = node_counts / node_counts.sum()
        edge_counts = edge_counts / edge_counts.sum()
        return node_counts, edge_counts

    def _build_model(self) -> None:
        node_marginals, edge_marginals = self._compute_marginals()
        self._transition_model = _MarginalUniformTransition(node_marginals, edge_marginals, y_classes=0)
        self._limit_dist = _PlaceHolder(
            X=node_marginals,
            E=edge_marginals,
            y=torch.ones(0, dtype=torch.float32),
        )
        self._noise_schedule = _PredefinedNoiseScheduleDiscrete("cosine", self._T)
        self._build_model_architecture()

    def _build_model_architecture(self) -> None:
        xdim = len(self._id_to_node_label)
        edim = len(self._id_to_edge_label)
        input_dims = {"X": xdim, "E": edim, "y": 1}
        output_dims = {"X": xdim, "E": edim, "y": 0}
        self._model = _GraphTransformer(
            n_layers=self._n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=self._hidden_mlp_dims,
            hidden_dims=self._hidden_dims,
            output_dims=output_dims,
        ).to(self._device)

    def _export_active_state(self) -> dict[str, Any]:
        return {
            "dataset": self._dataset,
            "node_dist": self._node_dist,
            "transition_model": self._transition_model,
            "limit_dist": self._limit_dist,
            "noise_schedule": self._noise_schedule,
            "checkpoint_path": None,
            "best_epoch": None,
            "best_val_loss": None,
        }

    def _load_active_state(self, state: dict[str, Any]) -> None:
        self._dataset = state["dataset"]
        self._node_dist = state["node_dist"]
        self._transition_model = state["transition_model"]
        self._limit_dist = state["limit_dist"]
        self._noise_schedule = state["noise_schedule"]
        self._build_model_architecture()
        checkpoint_path = state["checkpoint_path"]
        if checkpoint_path is None:
            raise RuntimeError("Missing checkpoint path for target state.")
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        if self._model is None:
            raise RuntimeError("Failed to rebuild model architecture.")
        self._model.load_state_dict(checkpoint["model_state_dict"])

    def _new_torch_generator(self, seed: int) -> torch.Generator:
        device_type = self._device.type if isinstance(self._device, torch.device) else str(self._device)
        generator = torch.Generator(device=device_type)
        generator.manual_seed(seed)
        return generator

    def _checkpoint_path_for_target(self, target_name: Any) -> str:
        if self._checkpoint_dir is None:
            self._checkpoint_dir = tempfile.mkdtemp(prefix="digress_graph_generator_")
        target_repr = "unconditional" if target_name is None else repr(target_name)
        target_hash = hashlib.md5(target_repr.encode("utf-8")).hexdigest()[:12]
        return os.path.join(self._checkpoint_dir, f"best_{target_hash}.pt")

    def _select_target_state(self, target: Any | None) -> None:
        if not self._class_states:
            raise RuntimeError("Generator is not fitted.")
        if len(self._class_states) == 1 and target is None:
            target = next(iter(self._class_states))
        elif len(self._class_states) > 1 and target is None:
            available = ", ".join(repr(key) for key in self._class_states)
            raise ValueError(f"target must be provided for class-conditional generation. Available targets: {available}")
        if target not in self._class_states:
            available = ", ".join(repr(key) for key in self._class_states)
            raise ValueError(f"Unknown target {target!r}. Available targets: {available}")
        self._active_target = target
        self._load_active_state(self._class_states[target])

    def _fit_single_dataset(
        self,
        dataset: list[dict[str, Any]],
        *,
        epochs: int,
        batch_size: int,
        lr: float,
        seed: int,
        target_name: Any,
    ) -> dict[str, Any]:
        if len(dataset) >= 2:
            indices = list(range(len(dataset)))
            shuffler = random.Random(seed)
            shuffler.shuffle(indices)
            val_size = max(1, int(round(0.1 * len(dataset))))
            train_size = len(dataset) - val_size
            if train_size == 0:
                train_size = len(dataset) - 1
                val_size = 1
            train_dataset = [dataset[idx] for idx in indices[:train_size]]
            val_dataset = [dataset[idx] for idx in indices[train_size:]]
        else:
            train_dataset = dataset
            val_dataset = []
        monitor_indices = list(range(len(train_dataset)))
        random.Random(seed + 17).shuffle(monitor_indices)
        monitor_indices = monitor_indices[: min(self._real_loss_subset_size, len(train_dataset))]
        monitor_dataset = [train_dataset[idx] for idx in monitor_indices]
        self._dataset = train_dataset
        _set_seed(seed)
        self._generator = self._new_torch_generator(seed)
        self._build_model()
        if self._model is None:
            raise RuntimeError("Failed to initialize the DiGress model.")
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr, amsgrad=True, weight_decay=1e-12)
        loader = DataLoader(
            _GraphDataset(train_dataset),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate,
            generator=self._generator,
        )
        monitor_loader = DataLoader(
            _GraphDataset(monitor_dataset),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate,
        )
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                _GraphDataset(val_dataset),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self._collate,
            )
        self._model.train()
        epoch_losses: list[float] = []
        real_losses: list[float] = []
        val_losses: list[float] = []
        best_val_loss = float("inf")
        checkpoint_path = self._checkpoint_path_for_target(target_name)
        torch.save(
            {
                "model_state_dict": copy.deepcopy(self._model.state_dict()),
                "target": target_name,
                "epoch": 0,
                "val_loss": best_val_loss,
                "seed": seed,
            },
            checkpoint_path,
        )
        best_epoch = 0
        epochs_without_improvement = 0
        for _ in range(epochs):
            running_loss = 0.0
            num_batches = 0
            for X, E, node_mask in loader:
                X = X.to(self._device)
                E = E.to(self._device)
                node_mask = node_mask.to(self._device)
                noisy_data = self.forward_transition(X, E, node_mask)
                pred = self._forward_model(noisy_data, node_mask)
                loss = self.compute_loss(pred, X, E)
                if torch.isnan(loss):
                    raise RuntimeError("NaN encountered in training loss.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.detach().cpu().item())
                num_batches += 1
            epoch_loss = running_loss / max(num_batches, 1)
            epoch_losses.append(epoch_loss)
            epoch_idx = len(epoch_losses)
            should_log_real_loss = (epoch_idx == 1) or (epoch_idx % self._real_loss_log_every == 0)
            real_train_loss = float("nan")
            if should_log_real_loss:
                real_train_loss = self._evaluate_real_loss(
                    monitor_loader,
                    seed=seed + 25_000,
                )
            real_losses.append(real_train_loss)
            if val_loader is not None:
                epoch_val_loss = self._evaluate_real_loss(
                    val_loader,
                    seed=seed + 50_000,
                )
                val_losses.append(epoch_val_loss)
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_epoch = len(epoch_losses)
                    epochs_without_improvement = 0
                    torch.save(
                        {
                            "model_state_dict": copy.deepcopy(self._model.state_dict()),
                            "target": target_name,
                            "epoch": best_epoch,
                            "val_loss": best_val_loss,
                            "seed": seed,
                        },
                        checkpoint_path,
                    )
                else:
                    epochs_without_improvement += 1
            else:
                epoch_val_loss = float("nan")
            if self._verbose:
                prefix = "train"
                if target_name is not None:
                    prefix = f"train[target={target_name!r}]"
                if val_loader is not None:
                    real_loss_display = f"{real_train_loss:.6f}" if not math.isnan(real_train_loss) else "skipped"
                    print(
                        f"{prefix} epoch={len(epoch_losses)}/{epochs} "
                        f"surrogate_loss={epoch_loss:.6f} "
                        f"real_loss={real_loss_display} "
                        f"val_real_loss={epoch_val_loss:.6f}"
                    )
                else:
                    real_loss_display = f"{real_train_loss:.6f}" if not math.isnan(real_train_loss) else "skipped"
                    print(
                        f"{prefix} epoch={len(epoch_losses)}/{epochs} "
                        f"surrogate_loss={epoch_loss:.6f} real_loss={real_loss_display}"
                    )
            if val_loader is not None and epochs_without_improvement >= self._early_stopping_patience:
                if self._verbose:
                    prefix = "train"
                    if target_name is not None:
                        prefix = f"train[target={target_name!r}]"
                    print(
                        f"{prefix} early_stop epoch={len(epoch_losses)} "
                        f"best_epoch={best_epoch} best_val_loss={best_val_loss:.6f}"
                    )
                break
        if val_loader is not None:
            best_checkpoint = torch.load(checkpoint_path, map_location=self._device)
            self._model.load_state_dict(best_checkpoint["model_state_dict"])
        else:
            best_epoch = len(epoch_losses)
            torch.save(
                {
                    "model_state_dict": copy.deepcopy(self._model.state_dict()),
                    "target": target_name,
                    "epoch": best_epoch,
                    "val_loss": None,
                    "seed": seed,
                },
                checkpoint_path,
            )
        self._loss_history[target_name] = epoch_losses
        self._real_loss_history[target_name] = real_losses
        self._val_loss_history[target_name] = val_losses
        self._dataset = dataset
        state = self._export_active_state()
        state["checkpoint_path"] = checkpoint_path
        state["best_epoch"] = best_epoch
        state["best_val_loss"] = best_val_loss if val_loader is not None else None
        return state

    def _evaluate_real_loss(
        self,
        loader: DataLoader,
        *,
        seed: int,
    ) -> float:
        """Evaluate the DiGress variational/NLL-style loss with deterministic corruption.

        Args:
            loader: DataLoader to evaluate.
            seed: Seed for deterministic corruption during evaluation.

        Returns:
            Average real loss over the loader.
        """
        self._model.eval()
        running_loss = 0.0
        num_batches = 0
        generator = self._new_torch_generator(seed)
        with torch.no_grad():
            for X, E, node_mask in loader:
                X = X.to(self._device)
                E = E.to(self._device)
                node_mask = node_mask.to(self._device)
                real_loss = self._compute_val_nll(X, E, node_mask, generator=generator)
                running_loss += float(real_loss.mean().detach().cpu().item())
                num_batches += 1
        self._model.train()
        return running_loss / max(num_batches, 1)

    def _compute_extra_data(self, noisy_data: dict[str, torch.Tensor]) -> _PlaceHolder:
        extra_features = self._extra_features(noisy_data)
        extra_domain = self._domain_features(noisy_data)
        extra_X = torch.cat((extra_features.X, extra_domain.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_domain.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_domain.y), dim=-1)
        t = noisy_data["t"]
        extra_y = torch.cat((extra_y, t), dim=1)
        return _PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    def _forward_model(self, noisy_data: dict[str, torch.Tensor], node_mask: torch.Tensor) -> _PlaceHolder:
        if self._model is None:
            raise RuntimeError("Model is not initialized.")
        extra_data = self._compute_extra_data(noisy_data)
        X = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data["y_t"], extra_data.y)).float()
        return self._model(X, E, y, node_mask)

    def forward_transition(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        node_mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Apply the DiGress forward categorical noising process.

        Args:
            X: One-hot node tensor `[bs, n, dx]`.
            E: One-hot edge tensor `[bs, n, n, de]`.
            node_mask: Boolean node mask `[bs, n]`.
            generator: Optional generator override.

        Returns:
            Dictionary containing `t`, `beta_t`, `alpha_s_bar`, `alpha_t_bar`,
            and the sampled noisy graph `X_t`, `E_t`.
        """
        if self._noise_schedule is None or self._transition_model is None:
            raise RuntimeError("Generator is not fitted.")
        lowest_t = 0 if self._model.training else 1
        t_int = torch.randint(
            lowest_t,
            self._T + 1,
            size=(X.size(0), 1),
            device=X.device,
            generator=generator if generator is not None else self._generator,
        ).float()
        s_int = t_int - 1
        t_float = t_int / self._T
        s_float = s_int / self._T
        beta_t = self._noise_schedule(t_normalized=t_float)
        alpha_s_bar = self._noise_schedule.get_alpha_bar(t_normalized=s_float)
        alpha_t_bar = self._noise_schedule.get_alpha_bar(t_normalized=t_float)
        Qtb = self._transition_model.get_Qt_bar(alpha_t_bar, device=self._device)
        probX = X @ Qtb.X
        probE = E @ Qtb.E.unsqueeze(1)
        sampled_t = _sample_discrete_features(
            probX,
            probE,
            node_mask=node_mask,
            generator=generator if generator is not None else self._generator,
        )
        X_t = F.one_hot(sampled_t.X, num_classes=len(self._id_to_node_label)).float()
        E_t = F.one_hot(sampled_t.E, num_classes=len(self._id_to_edge_label)).float()
        z_t = _PlaceHolder(X=X_t, E=E_t, y=X_t.new_zeros(X_t.size(0), 0)).mask(node_mask)
        return {
            "t_int": t_int,
            "t": t_float,
            "beta_t": beta_t,
            "alpha_s_bar": alpha_s_bar,
            "alpha_t_bar": alpha_t_bar,
            "X_t": z_t.X,
            "E_t": z_t.E,
            "y_t": z_t.y,
            "node_mask": node_mask,
        }

    def compute_loss(
        self,
        pred: _PlaceHolder,
        true_X: torch.Tensor,
        true_E: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the official DiGress discrete training loss.

        This matches `src/metrics/train_metrics.py:TrainLossDiscrete.forward`.

        Args:
            pred: Predicted logits.
            true_X: True one-hot node classes.
            true_E: True one-hot edge classes.

        Returns:
            Scalar training loss.
        """
        true_X_flat = true_X.reshape(-1, true_X.size(-1))
        true_E_flat = true_E.reshape(-1, true_E.size(-1))
        pred_X_flat = pred.X.reshape(-1, pred.X.size(-1))
        pred_E_flat = pred.E.reshape(-1, pred.E.size(-1))
        mask_X = (true_X_flat != 0.0).any(dim=-1)
        mask_E = (true_E_flat != 0.0).any(dim=-1)
        flat_true_X = true_X_flat[mask_X]
        flat_true_E = true_E_flat[mask_E]
        flat_pred_X = pred_X_flat[mask_X]
        flat_pred_E = pred_E_flat[mask_E]
        loss_X = F.cross_entropy(flat_pred_X, flat_true_X.argmax(dim=-1), reduction="mean")
        loss_E = F.cross_entropy(flat_pred_E, flat_true_E.argmax(dim=-1), reduction="mean")
        return loss_X + self._lambda_train[0] * loss_E

    def _kl_prior(self, X: torch.Tensor, E: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        if self._noise_schedule is None or self._transition_model is None or self._limit_dist is None:
            raise RuntimeError("Generator is not fitted.")
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self._T * ones
        alpha_t_bar = self._noise_schedule.get_alpha_bar(t_int=Ts)
        Qtb = self._transition_model.get_Qt_bar(alpha_t_bar, self._device)
        probX = X @ Qtb.X
        probE = E @ Qtb.E.unsqueeze(1)
        bs, n, _ = probX.shape
        limit_X = self._limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self._limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)
        limit_X, limit_E, probX, probE = _mask_distributions(limit_X, limit_E, probX, probE, node_mask)
        kl_x = F.kl_div(probX.log(), limit_X, reduction="none")
        kl_e = F.kl_div(probE.log(), limit_E, reduction="none")
        return _sum_except_batch(kl_x) + _sum_except_batch(kl_e)

    def _compute_Lt(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        pred: _PlaceHolder,
        noisy_data: dict[str, torch.Tensor],
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self._transition_model is None:
            raise RuntimeError("Generator is not fitted.")
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        Qtb = self._transition_model.get_Qt_bar(noisy_data["alpha_t_bar"], self._device)
        Qsb = self._transition_model.get_Qt_bar(noisy_data["alpha_s_bar"], self._device)
        Qt = self._transition_model.get_Qt(noisy_data["beta_t"], self._device)
        bs, n, _ = X.shape
        prob_true = _posterior_distributions(
            X=X,
            E=E,
            y=noisy_data["y_t"],
            X_t=noisy_data["X_t"],
            E_t=noisy_data["E_t"],
            y_t=noisy_data["y_t"],
            Qt=Qt,
            Qsb=Qsb,
            Qtb=Qtb,
        )
        prob_true.E = prob_true.E.reshape(bs, n, n, -1)
        prob_pred = _posterior_distributions(
            X=pred_probs_X,
            E=pred_probs_E,
            y=noisy_data["y_t"],
            X_t=noisy_data["X_t"],
            E_t=noisy_data["E_t"],
            y_t=noisy_data["y_t"],
            Qt=Qt,
            Qsb=Qsb,
            Qtb=Qtb,
        )
        prob_pred.E = prob_pred.E.reshape(bs, n, n, -1)
        true_X, true_E, pred_X, pred_E = _mask_distributions(
            prob_true.X, prob_true.E, prob_pred.X, prob_pred.E, node_mask
        )
        kl_x = F.kl_div(pred_X.log(), true_X, reduction="none").sum(dim=(-1, -2))
        kl_e = F.kl_div(pred_E.log(), true_E, reduction="none").sum(dim=(-1, -2, -3))
        return self._T * (kl_x + kl_e)

    def _reconstruction_logp(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        node_mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> _PlaceHolder:
        if self._noise_schedule is None or self._transition_model is None:
            raise RuntimeError("Generator is not fitted.")
        t_zeros = torch.zeros(X.shape[0], 1, device=X.device)
        beta_0 = self._noise_schedule(t_zeros)
        Q0 = self._transition_model.get_Qt(beta_0, self._device)
        probX0 = X @ Q0.X
        probE0 = E @ Q0.E.unsqueeze(1)
        sampled0 = _sample_discrete_features(
            probX0,
            probE0,
            node_mask=node_mask,
            generator=generator if generator is not None else self._generator,
        )
        X0 = F.one_hot(sampled0.X, num_classes=len(self._id_to_node_label)).float()
        E0 = F.one_hot(sampled0.E, num_classes=len(self._id_to_edge_label)).float()
        sampled_0 = _PlaceHolder(X=X0, E=E0, y=X0.new_zeros(X0.size(0), 0)).mask(node_mask)
        noisy_data = {
            "X_t": sampled_0.X,
            "E_t": sampled_0.E,
            "y_t": sampled_0.y,
            "node_mask": node_mask,
            "t": torch.zeros(X0.shape[0], 1, device=X0.device),
        }
        pred0 = self._forward_model(noisy_data, node_mask)
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        probX0 = probX0.clone()
        probE0 = probE0.clone()
        probX0[~node_mask] = torch.ones(len(self._id_to_node_label), device=X0.device)
        valid_edges = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        probE0[~valid_edges] = torch.ones(len(self._id_to_edge_label), device=X0.device)
        diag_mask = torch.eye(probE0.size(1), device=probE0.device, dtype=torch.bool).unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(len(self._id_to_edge_label), device=X0.device)
        return _PlaceHolder(X=probX0, E=probE0, y=probX0.new_zeros(probX0.size(0), 0))

    def _compute_val_nll(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        node_mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        noisy_data = self.forward_transition(X, E, node_mask, generator=generator)
        pred = self._forward_model(noisy_data, node_mask)
        N = node_mask.sum(1).long()
        if self._node_dist is None:
            raise RuntimeError("Generator is not fitted.")
        log_pN = self._node_dist.log_prob(N)
        kl_prior = self._kl_prior(X, E, node_mask)
        loss_all_t = self._compute_Lt(X, E, pred, noisy_data, node_mask)
        prob0 = self._reconstruction_logp(X, E, node_mask, generator=generator)
        loss_term_0 = (X * prob0.X.log()).sum(dim=(-1, -2)) + (E * prob0.E.log()).sum(dim=(-1, -2, -3))
        return -log_pN + kl_prior + loss_all_t - loss_term_0

    def sample_reverse(
        self,
        n_samples: int,
        *,
        num_nodes: int | None = None,
        seed: int | None = None,
        target: Any | None = None,
    ) -> list[nx.Graph]:
        """Sample graphs from the reverse DiGress process.

        Args:
            n_samples: Number of graphs.
            num_nodes: Fixed node count. If `None`, sample from the empirical
                node-count distribution learned in `fit`.
            seed: Optional sampling seed.
            target: Class label to sample from when `fit(..., targets=...)`
                was used.

        Returns:
            Generated NetworkX graphs.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before generate().")
        self._select_target_state(target)
        if seed is None:
            seed = self._train_seed + self._sample_seed_offset
        _set_seed(seed)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)
        if self._node_dist is None or self._limit_dist is None:
            raise RuntimeError("Generator is not fitted.")
        if num_nodes is None:
            n_nodes = self._node_dist.sample_n(n_samples, self._device)
        else:
            if num_nodes <= 0 or num_nodes > self._MAX_NODES:
                raise ValueError(f"num_nodes must be in [1, {self._MAX_NODES}].")
            n_nodes = torch.full((n_samples,), num_nodes, device=self._device, dtype=torch.long)
        n_max = int(torch.max(n_nodes).item())
        arange = torch.arange(n_max, device=self._device).unsqueeze(0).expand(n_samples, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        z_T = _sample_discrete_feature_noise(self._limit_dist, node_mask, generator=self._generator)
        X, E, y = z_T.X.to(self._device), z_T.E.to(self._device), z_T.y.to(self._device)
        if self._model is None or self._transition_model is None or self._noise_schedule is None:
            raise RuntimeError("Generator is not fitted.")
        self._model.eval()
        with torch.no_grad():
            for s_int in reversed(range(0, self._T)):
                s_array = torch.full((n_samples, 1), float(s_int), device=self._device)
                t_array = s_array + 1
                s_norm = s_array / self._T
                t_norm = t_array / self._T
                sampled_s, _ = self._sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask)
                X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
            sampled = sampled_s.mask(node_mask, collapse=True)
        graphs: list[nx.Graph] = []
        for i in range(n_samples):
            n = int(n_nodes[i].item())
            graphs.append(self._decode_graph(sampled.X[i, :n].cpu(), sampled.E[i, :n, :n].cpu()))
        return graphs

    def _sample_p_zs_given_zt(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        X_t: torch.Tensor,
        E_t: torch.Tensor,
        y_t: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[_PlaceHolder, _PlaceHolder]:
        if self._noise_schedule is None or self._transition_model is None:
            raise RuntimeError("Generator is not fitted.")
        bs, n, _ = X_t.shape
        beta_t = self._noise_schedule(t_normalized=t)
        alpha_s_bar = self._noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self._noise_schedule.get_alpha_bar(t_normalized=t)
        Qtb = self._transition_model.get_Qt_bar(alpha_t_bar, self._device)
        Qsb = self._transition_model.get_Qt_bar(alpha_s_bar, self._device)
        Qt = self._transition_model.get_Qt(beta_t, self._device)
        noisy_data = {"X_t": X_t, "E_t": E_t, "y_t": y_t, "t": t, "node_mask": node_mask}
        pred = self._forward_model(noisy_data, node_mask)
        pred_X = F.softmax(pred.X, dim=-1)
        pred_E = F.softmax(pred.E, dim=-1)
        p_s_and_t_given_0_X = _compute_batched_over0_posterior_distribution(X_t=X_t, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X)
        p_s_and_t_given_0_E = _compute_batched_over0_posterior_distribution(X_t=E_t, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E)
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X
        unnormalized_prob_X = weighted_X.sum(dim=2)
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)
        pred_E_flat = pred_E.reshape(bs, -1, pred_E.shape[-1])
        weighted_E = pred_E_flat.unsqueeze(-1) * p_s_and_t_given_0_E
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])
        sampled_s = _sample_discrete_features(prob_X, prob_E, node_mask=node_mask, generator=self._generator)
        X_s = F.one_hot(sampled_s.X, num_classes=len(self._id_to_node_label)).float()
        E_s = F.one_hot(sampled_s.E, num_classes=len(self._id_to_edge_label)).float()
        out_one_hot = _PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0, device=y_t.device))
        out_discrete = _PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0, device=y_t.device))
        return out_one_hot.mask(node_mask).type_as(X_t), out_discrete.mask(node_mask, collapse=True).type_as(X_t)

    def _decode_graph(self, node_ids: torch.Tensor, edge_ids: torch.Tensor) -> nx.Graph:
        graph = nx.Graph()
        for idx, node_id in enumerate(node_ids.tolist()):
            graph.add_node(idx, **{self._node_label_key: self._id_to_node_label[node_id]})
        n = len(node_ids)
        for i in range(n):
            for j in range(i + 1, n):
                edge_id = int(edge_ids[i, j].item())
                if edge_id != 0:
                    graph.add_edge(i, j, **{self._edge_label_key: self._id_to_edge_label[edge_id]})
        return graph

    def fit(
        self,
        graphs: list[nx.Graph],
        targets: list[Any] | None = None,
        *,
        node_label_key: str = "label",
        edge_label_key: str = "label",
        epochs: int = 10,
        batch_size: int = 16,
        lr: float = 2e-4,
        device: str | torch.device | None = None,
        seed: int = 0,
        verbose: bool = False,
    ) -> "DiGressGraphGenerator":
        """Fit the DiGress generator on a list of labeled graphs.

        Args:
            graphs: Training graphs.
            targets: Optional graph-level class labels. When provided, the
                generator trains one separate DiGress model per target value.
            node_label_key: Node label attribute key.
            edge_label_key: Edge label attribute key.
            epochs: Number of training epochs.
            batch_size: Batch size.
            lr: AdamW learning rate.
            device: Torch device. Defaults to CPU unless CUDA is available and
                explicitly requested.
            seed: Random seed.
            verbose: If true, print per-epoch training and validation loss.

        Returns:
            Self. Training uses an internal validation split when possible,
            early stopping with patience 5, and restores the best validation
            checkpoint in memory before returning.
        """
        if not graphs:
            raise ValueError("graphs must be a non-empty list.")
        self._node_label_key = node_label_key
        self._edge_label_key = edge_label_key
        self._train_seed = seed
        self._verbose = verbose
        self._loss_history = {}
        self._real_loss_history = {}
        self._val_loss_history = {}
        _set_seed(seed)
        self._device = torch.device(device if device is not None else "cpu")
        for graph in graphs:
            self._validate_graph(graph)
        if targets is not None and len(targets) != len(graphs):
            raise ValueError("targets must have the same length as graphs.")
        self._build_vocabularies(graphs)
        encoded_graphs = [self._encode_graph(graph) for graph in graphs]
        self._class_states = {}
        if targets is None:
            self._class_states[None] = self._fit_single_dataset(
                encoded_graphs,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                seed=seed,
                target_name=None,
            )
        else:
            ordered_targets = sorted(set(targets), key=lambda value: (type(value).__name__, repr(value)))
            for offset, target in enumerate(ordered_targets):
                class_dataset = [sample for sample, sample_target in zip(encoded_graphs, targets) if sample_target == target]
                if not class_dataset:
                    continue
                self._class_states[target] = self._fit_single_dataset(
                    class_dataset,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    seed=seed + offset,
                    target_name=target,
                )
        self._select_target_state(next(iter(self._class_states)))
        self._is_fitted = True
        return self

    def generate(
        self,
        n_samples: int,
        *,
        num_nodes: int | None = None,
        seed: int | None = None,
        target: Any | None = None,
    ) -> list[nx.Graph]:
        """Generate graphs with the fitted DiGress reverse process.

        Args:
            n_samples: Number of graphs to generate.
            num_nodes: Fixed node count. If omitted, sample from the empirical
                node-count distribution.
            seed: Optional sampling seed.
            target: Class label to sample from when `fit(..., targets=...)`
                was used.

        Returns:
            Generated NetworkX graphs.
        """
        return self.sample_reverse(n_samples, num_nodes=num_nodes, seed=seed, target=target)


if __name__ == "__main__":
    _set_seed(7)

    def _make_synthetic_dataset(n_graphs: int = 30) -> tuple[list[nx.Graph], list[int]]:
        graphs_out: list[nx.Graph] = []
        targets_out: list[int] = []
        rng = np.random.default_rng(7)
        for i in range(n_graphs):
            n = int(rng.integers(4, 8))
            is_cycle = i % 2 == 0
            graph = nx.cycle_graph(n) if is_cycle else nx.path_graph(n)
            targets_out.append(1 if is_cycle else 0)
            extra_edges = int(rng.integers(0, max(1, n // 2)))
            for _ in range(extra_edges):
                u = int(rng.integers(0, n))
                v = int(rng.integers(0, n))
                if u != v:
                    graph.add_edge(u, v)
            for node in graph.nodes():
                graph.nodes[node]["label"] = "A" if (node + i) % 2 == 0 else "B"
            for u, v in graph.edges():
                graph.edges[u, v]["label"] = "x" if (u + v + i) % 2 == 0 else "y"
            graphs_out.append(graph)
        return graphs_out, targets_out

    dataset, targets = _make_synthetic_dataset()
    generator = DiGressGraphGenerator().fit(dataset, epochs=1, batch_size=8, lr=2e-4, device="cpu", seed=7)
    samples = generator.generate(3, seed=11)
    assert len(samples) == 3
    for graph in samples:
        assert isinstance(graph, nx.Graph)
        for _, attrs in graph.nodes(data=True):
            assert "label" in attrs
        for _, _, attrs in graph.edges(data=True):
            assert "label" in attrs
    avg_nodes = sum(graph.number_of_nodes() for graph in samples) / len(samples)
    avg_edges = sum(graph.number_of_edges() for graph in samples) / len(samples)
    conditional_generator = DiGressGraphGenerator().fit(
        dataset,
        targets=targets,
        epochs=1,
        batch_size=8,
        lr=2e-4,
        device="cpu",
        seed=13,
    )
    conditional_samples = conditional_generator.generate(2, target=1, seed=17)
    assert len(conditional_samples) == 2
    print(
        f"Smoke test passed. Generated {len(samples)} unconditional and "
        f"{len(conditional_samples)} conditional graphs. avg_nodes={avg_nodes:.2f}, avg_edges={avg_edges:.2f}"
    )
