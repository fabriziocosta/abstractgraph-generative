"""Self-contained VGAE graph generator for NetworkX graphs."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


def _sparse_to_tuple(sparse_mx: sp.spmatrix) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Convert a SciPy sparse matrix to tuple representation.

    Args:
        sparse_mx: SciPy sparse matrix.

    Returns:
        Tuple of coordinates, values, and shape.
    """
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def _preprocess_graph(adj: sp.spmatrix) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Compute normalized adjacency D^(-1/2)(A+I)D^(-1/2).

    Args:
        adj: Sparse adjacency matrix.

    Returns:
        Normalized adjacency in sparse tuple format.
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0], dtype=np.float32)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return _sparse_to_tuple(adj_normalized)


def _to_torch_sparse(sparse_tuple: Tuple[np.ndarray, np.ndarray, Tuple[int, int]]) -> torch.Tensor:
    """
    Build sparse torch tensor from sparse tuple.

    Args:
        sparse_tuple: Tuple format sparse matrix.

    Returns:
        Sparse COO torch tensor.
    """
    coords, values, shape = sparse_tuple
    indices = torch.LongTensor(coords.T)
    vals = torch.FloatTensor(values)
    return torch.sparse_coo_tensor(indices, vals, torch.Size(shape)).coalesce()


def _glorot_init(input_dim: int, output_dim: int) -> nn.Parameter:
    """
    Glorot initialization for trainable weight matrix.

    Args:
        input_dim: Input dimension.
        output_dim: Output dimension.

    Returns:
        Parameter tensor with Glorot uniform initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


def _dot_product_decode(z: torch.Tensor) -> torch.Tensor:
    """
    Decode latent embeddings to adjacency probabilities.

    Args:
        z: Latent node embeddings.

    Returns:
        Dense adjacency probability matrix.
    """
    return torch.sigmoid(torch.matmul(z, z.t()))


class _GraphConvSparse(nn.Module):
    """Sparse graph convolution layer."""

    def __init__(self, input_dim: int, output_dim: int, activation=F.relu) -> None:
        """
        Initialize sparse GCN layer.

        Args:
            input_dim: Input feature dimension.
            output_dim: Output feature dimension.
            activation: Activation function.

        Returns:
            None.
        """
        super().__init__()
        self.weight = _glorot_init(input_dim, output_dim)
        self.activation = activation

    def forward(self, inputs: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass.

        Args:
            inputs: Dense node feature matrix.
            adj_norm: Normalized sparse adjacency matrix.

        Returns:
            Output embedding matrix.
        """
        x = torch.mm(inputs, self.weight)
        x = torch.mm(adj_norm, x)
        return self.activation(x)


class _VGAEModel(nn.Module):
    """Variational Graph Autoencoder."""

    def __init__(
        self,
        input_dim: int,
        hidden1_dim: int,
        hidden2_dim: int,
        *,
        dropout_rate: float = 0.1,
        encoder_num_layers: int = 2,
        use_residual: bool = True,
    ) -> None:
        """
        Initialize VGAE model.

        Args:
            input_dim: Input feature dimension.
            hidden1_dim: Hidden dimension.
            hidden2_dim: Latent dimension.
            dropout_rate: Dropout probability applied in encoder hidden representation.
            encoder_num_layers: Number of hidden graph-convolution blocks before latent heads.
            use_residual: If True, apply residual connections across hidden blocks.

        Returns:
            None.
        """
        super().__init__()
        self.hidden2_dim = hidden2_dim
        self.encoder_num_layers = max(1, int(encoder_num_layers))
        self.use_residual = bool(use_residual)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.hidden_norms = nn.ModuleList([nn.LayerNorm(hidden1_dim) for _ in range(self.encoder_num_layers)])
        self.mean_norm = nn.LayerNorm(hidden2_dim)
        self.logstd_norm = nn.LayerNorm(hidden2_dim)
        self.input_gcn = _GraphConvSparse(input_dim, hidden1_dim)
        self.hidden_gcns = nn.ModuleList(
            [_GraphConvSparse(hidden1_dim, hidden1_dim) for _ in range(self.encoder_num_layers - 1)]
        )
        self.gcn_mean = _GraphConvSparse(hidden1_dim, hidden2_dim, activation=lambda x: x)
        self.gcn_logstddev = _GraphConvSparse(hidden1_dim, hidden2_dim, activation=lambda x: x)
        self.mean: Optional[torch.Tensor] = None
        self.logstd: Optional[torch.Tensor] = None

    def encode(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Encode features to latent variables.

        Args:
            x: Dense node features.
            adj_norm: Normalized sparse adjacency matrix.

        Returns:
            Sampled latent embeddings.
        """
        hidden = self.input_gcn(x, adj_norm)
        hidden = self.hidden_norms[0](hidden)
        hidden = self.dropout(hidden)
        for layer_idx, gcn in enumerate(self.hidden_gcns, start=1):
            residual = hidden
            hidden = gcn(hidden, adj_norm)
            hidden = self.hidden_norms[layer_idx](hidden)
            hidden = self.dropout(hidden)
            if self.use_residual:
                hidden = hidden + residual
        self.mean = self.mean_norm(self.gcn_mean(hidden, adj_norm))
        self.logstd = self.logstd_norm(self.gcn_logstddev(hidden, adj_norm))
        eps = torch.randn(x.size(0), self.hidden2_dim)
        return eps * torch.exp(self.logstd) + self.mean

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Run VGAE forward pass.

        Args:
            x: Dense node features.
            adj_norm: Normalized sparse adjacency matrix.

        Returns:
            Predicted adjacency probabilities.
        """
        z = self.encode(x, adj_norm)
        return _dot_product_decode(z)


class VGAEGraphGenerator:
    """Fit a self-contained VGAE on NetworkX graphs and sample similar graphs."""

    def __init__(
        self,
        *,
        hidden1_dim: int = 32,
        hidden2_dim: int = 16,
        encoder_num_layers: int = 3,
        use_residual: bool = True,
        learning_rate: float = 1e-3,
        num_epochs: int = 500,
        edge_threshold: float = 0.5,
        calibrate_edge_threshold: bool = True,
        sample_edges_bernoulli: bool = False,
        dropout_rate: float = 0.15,
        validation_split: float = 0.15,
        early_stopping_patience: int = 5,
        early_stopping_min_delta: float = 1e-4,
        restore_best_checkpoint: bool = True,
        validation_strategy: str = "holdout",
        validation_repeats: int = 3,
        validation_k_folds: int = 3,
        ranking_loss_weight: float = 0.1,
        ranking_num_negative: int = 512,
        calibration_quantile_smoothing: float = 0.5,
        calibration_min_sample_size: int = 256,
        calibration_fallback_to_previous: bool = True,
        dynamic_density_matching: bool = True,
        density_match_blend: float = 1.0,
        enforce_connected: bool = True,
        connectivity_threshold_step: float = 0.02,
        connectivity_min_threshold: float = 0.01,
        connectivity_max_iters: int = 50,
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        beta_warmup_epochs: int = 100,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the generator.

        Args:
            hidden1_dim: Hidden size of first GCN layer.
            hidden2_dim: Latent embedding dimension.
            encoder_num_layers: Number of hidden graph-convolution blocks before latent heads.
            use_residual: If True, apply residual connections across hidden blocks.
            learning_rate: Optimizer learning rate.
            num_epochs: Maximum number of fit epochs (early stopping may stop earlier).
            edge_threshold: Threshold used to binarize sampled adjacency probabilities.
            calibrate_edge_threshold: If True, calibrate threshold per type to match validation edge density.
            sample_edges_bernoulli: If True, sample edges from Bernoulli probabilities instead of thresholding.
            dropout_rate: Encoder dropout probability.
            validation_split: Fraction of training graphs reserved for validation per type.
            early_stopping_patience: Number of epochs without improvement before stopping.
            early_stopping_min_delta: Minimum validation loss improvement to reset patience.
            restore_best_checkpoint: If True, restore model weights from best validation epoch.
            validation_strategy: One of {'holdout', 'repeated_holdout', 'kfold'}.
            validation_repeats: Number of splits for repeated holdout.
            validation_k_folds: Number of folds for k-fold strategy.
            ranking_loss_weight: Weight of ranking contrastive term added to BCE+KL objective.
            ranking_num_negative: Max number of sampled negative edges per graph for ranking loss.
            calibration_quantile_smoothing: Blend factor between previous and newly calibrated threshold.
            calibration_min_sample_size: Minimum number of validation edge samples required for calibration.
            calibration_fallback_to_previous: If True, keep previous threshold when calibration is noisy.
            dynamic_density_matching: If True, adapt threshold per sampled graph to match learned edge density.
            density_match_blend: Blend between fixed threshold and dynamic density-matched threshold (0..1).
            enforce_connected: If True, adapt threshold downward until generated graph is connected.
            connectivity_threshold_step: Threshold decrement per connectivity iteration.
            connectivity_min_threshold: Lower bound threshold used during connectivity enforcement.
            connectivity_max_iters: Max iterations for threshold descent before fallback connection.
            beta_start: Initial KL weight for beta-VAE schedule.
            beta_end: Final KL weight for beta-VAE schedule.
            beta_warmup_epochs: Number of warmup epochs for beta schedule.
            random_state: Base random seed.
            verbose: If True, print training progress and loss during fit.

        Returns:
            None.
        """
        self.hidden1_dim = int(hidden1_dim)
        self.hidden2_dim = int(hidden2_dim)
        self.encoder_num_layers = int(encoder_num_layers)
        self.use_residual = bool(use_residual)
        self.learning_rate = float(learning_rate)
        self.num_epochs = int(num_epochs)
        self.edge_threshold = float(edge_threshold)
        self.calibrate_edge_threshold = bool(calibrate_edge_threshold)
        self.sample_edges_bernoulli = bool(sample_edges_bernoulli)
        self.dropout_rate = float(dropout_rate)
        self.validation_split = float(validation_split)
        self.early_stopping_patience = int(early_stopping_patience)
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.restore_best_checkpoint = bool(restore_best_checkpoint)
        self.validation_strategy = str(validation_strategy)
        self.validation_repeats = int(validation_repeats)
        self.validation_k_folds = int(validation_k_folds)
        self.ranking_loss_weight = float(ranking_loss_weight)
        self.ranking_num_negative = int(ranking_num_negative)
        self.calibration_quantile_smoothing = float(calibration_quantile_smoothing)
        self.calibration_min_sample_size = int(calibration_min_sample_size)
        self.calibration_fallback_to_previous = bool(calibration_fallback_to_previous)
        self.dynamic_density_matching = bool(dynamic_density_matching)
        self.density_match_blend = float(density_match_blend)
        self.enforce_connected = bool(enforce_connected)
        self.connectivity_threshold_step = float(connectivity_threshold_step)
        self.connectivity_min_threshold = float(connectivity_min_threshold)
        self.connectivity_max_iters = int(connectivity_max_iters)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_warmup_epochs = int(beta_warmup_epochs)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

        self._is_fitted = False
        self._label_feature_dim = 0
        self._states: Dict[Hashable, Dict[str, Any]] = {}

    def _set_random_seeds(self, seed: int) -> None:
        """
        Set NumPy and Torch random seeds.

        Args:
            seed: Seed value.

        Returns:
            None.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _infer_label_feature_dim(self, graphs: Sequence[nx.Graph]) -> int:
        """
        Infer one-hot feature width from max node label across all graphs.

        Args:
            graphs: Input graph list.

        Returns:
            Number of label feature dimensions.
        """
        max_label = 0
        for graph in graphs:
            if graph.number_of_nodes() == 0:
                continue
            labels = [int(graph.nodes[u].get("label", 0)) for u in graph.nodes()]
            if labels:
                max_label = max(max_label, max(labels))
        return max_label + 1

    def _bucket_graphs_by_target(
        self,
        graphs: Sequence[nx.Graph],
        targets: Sequence[Hashable],
    ) -> Dict[Hashable, List[nx.Graph]]:
        """
        Group graphs by target label.

        Args:
            graphs: Input graph list.
            targets: Per-graph type labels.

        Returns:
            Dict mapping target -> graphs.
        """
        buckets: Dict[Hashable, List[nx.Graph]] = {}
        for graph, target in zip(graphs, targets):
            buckets.setdefault(target, []).append(graph)
        return buckets

    def _split_train_val_graphs(
        self,
        graphs: Sequence[nx.Graph],
        *,
        seed: int,
        split_id: int = 0,
    ) -> Tuple[List[nx.Graph], List[nx.Graph]]:
        """
        Split graphs into train/validation subsets.

        Args:
            graphs: Input graphs for one type bucket.
            seed: Split seed.
            split_id: Additional split index to vary repeated holdout.

        Returns:
            Tuple of (train_graphs, val_graphs).
        """
        graphs = list(graphs)
        n = len(graphs)
        if n <= 1 or self.validation_split <= 0.0:
            return graphs, []

        rng = np.random.default_rng(seed + split_id)
        order = np.arange(n)
        rng.shuffle(order)

        n_val = int(round(self.validation_split * n))
        n_val = max(1, min(n - 1, n_val))

        val_idx = set(order[:n_val].tolist())
        train_graphs = [g for i, g in enumerate(graphs) if i not in val_idx]
        val_graphs = [g for i, g in enumerate(graphs) if i in val_idx]
        return train_graphs, val_graphs

    def _iter_train_val_splits(
        self,
        graphs: Sequence[nx.Graph],
        *,
        seed: int,
    ) -> List[Tuple[List[nx.Graph], List[nx.Graph]]]:
        """
        Build one or more train/validation splits according to strategy.

        Args:
            graphs: Input graphs for one type bucket.
            seed: Base split seed.

        Returns:
            List of (train_graphs, val_graphs) splits.
        """
        graphs = list(graphs)
        n = len(graphs)
        if n <= 1 or self.validation_split <= 0.0:
            return [(graphs, [])]

        strategy = self.validation_strategy.lower()
        if strategy == "holdout":
            return [self._split_train_val_graphs(graphs, seed=seed, split_id=0)]
        if strategy == "repeated_holdout":
            repeats = max(1, self.validation_repeats)
            return [self._split_train_val_graphs(graphs, seed=seed, split_id=i) for i in range(repeats)]
        if strategy == "kfold":
            k = max(2, min(self.validation_k_folds, n))
            rng = np.random.default_rng(seed)
            order = np.arange(n)
            rng.shuffle(order)
            folds = np.array_split(order, k)
            out: List[Tuple[List[nx.Graph], List[nx.Graph]]] = []
            for fold_idx in range(k):
                val_idx = set(folds[fold_idx].tolist())
                train_graphs = [g for i, g in enumerate(graphs) if i not in val_idx]
                val_graphs = [g for i, g in enumerate(graphs) if i in val_idx]
                if len(train_graphs) == 0:
                    continue
                out.append((train_graphs, val_graphs))
            return out if out else [(graphs, [])]
        raise ValueError(f"Unknown validation_strategy '{self.validation_strategy}'.")

    def _build_raw_features(
        self,
        graph: nx.Graph,
        *,
        n_label_features: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build raw dense node features and label vector for one graph.

        Features are one-hot labels concatenated with structural features:
        normalized degree, clustering coefficient, and betweenness centrality.

        Args:
            graph: Input graph.
            n_label_features: Number of one-hot label dimensions.

        Returns:
            Tuple of (raw_features, labels).
        """
        n_nodes = graph.number_of_nodes()
        labels = np.array([int(graph.nodes[u].get("label", 0)) for u in range(n_nodes)], dtype=np.int64)
        labels = np.clip(labels, 0, n_label_features - 1)

        label_one_hot = np.eye(n_label_features, dtype=np.float32)[labels]
        degrees = np.array([graph.degree(u) for u in range(n_nodes)], dtype=np.float32)
        degree_norm = degrees / max(float(max(n_nodes - 1, 1)), 1.0)
        clustering = np.array([nx.clustering(graph, u) for u in range(n_nodes)], dtype=np.float32)
        centrality_dict = nx.betweenness_centrality(graph, normalized=True)
        centrality = np.array([float(centrality_dict[u]) for u in range(n_nodes)], dtype=np.float32)
        structural = np.stack([degree_norm, clustering, centrality], axis=1).astype(np.float32)
        features = np.concatenate([label_one_hot, structural], axis=1).astype(np.float32)
        return features, labels

    def _fit_feature_scaler(
        self,
        graphs: Sequence[nx.Graph],
        *,
        n_label_features: int,
    ) -> Dict[str, Any]:
        """
        Fit per-feature standardization parameters on training graphs only.

        Args:
            graphs: Training graphs.
            n_label_features: Number of one-hot label dimensions.

        Returns:
            Dict with scaler metadata and structural feature normalization arrays.
        """
        feats = [self._build_raw_features(g, n_label_features=n_label_features)[0] for g in graphs]
        all_features = np.vstack(feats).astype(np.float32)
        # Keep one-hot label block unchanged; scale only continuous structural features.
        structural_start = int(n_label_features)
        structural = all_features[:, structural_start:]
        mean = structural.mean(axis=0)
        std = structural.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return {
            "n_label_features": int(n_label_features),
            "structural_start": structural_start,
            "structural_mean": mean.astype(np.float32),
            "structural_std": std,
        }

    @staticmethod
    def _apply_feature_scaler(features: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
        """
        Apply standardization to feature matrix.

        Args:
            features: Raw dense feature matrix.
            scaler: Dict containing `mean` and `std`.

        Returns:
            Feature matrix with one-hot block untouched and structural block standardized.
        """
        out = features.copy()
        start = int(scaler["structural_start"])
        out[:, start:] = (out[:, start:] - scaler["structural_mean"]) / scaler["structural_std"]
        return out

    def _make_graph_pack(
        self,
        graph: nx.Graph,
        *,
        scaler: Dict[str, Any],
        n_label_features: int,
    ) -> Dict[str, Any]:
        """
        Prepare one graph tensor pack for training/evaluation.

        Args:
            graph: Input graph.
            scaler: Feature scaler fitted on train graphs.
            n_label_features: Number of one-hot label dimensions.

        Returns:
            Dict containing tensors and metadata for one graph.
        """
        adjacency = nx.to_scipy_sparse_array(graph, dtype=np.float32, format="csr")
        features_raw, labels = self._build_raw_features(graph, n_label_features=n_label_features)
        features_np = self._apply_feature_scaler(features_raw, scaler)

        adj_norm_t = _to_torch_sparse(_preprocess_graph(adjacency))
        adj_label_t = _to_torch_sparse(
            _sparse_to_tuple(adjacency + sp.eye(adjacency.shape[0], dtype=np.float32, format="csr"))
        ).to_dense()
        features_t = torch.FloatTensor(features_np)

        total = adjacency.shape[0] * adjacency.shape[0]
        pos = float(adjacency.sum())
        pos_weight = float(total - pos) / max(pos, 1.0)
        norm = total / float((total - pos) * 2)
        weight_mask = adj_label_t.view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0))
        weight_tensor[weight_mask] = pos_weight

        # Cache upper-triangular positive and negative edge indices for ranking loss.
        dense_adj = adjacency.toarray().astype(np.int64)
        triu = np.triu_indices(adjacency.shape[0], k=1)
        triu_pos_mask = dense_adj[triu] == 1
        triu_neg_mask = ~triu_pos_mask
        pos_edges_upper = np.vstack((triu[0][triu_pos_mask], triu[1][triu_pos_mask])).T.astype(np.int64)
        neg_edges_upper = np.vstack((triu[0][triu_neg_mask], triu[1][triu_neg_mask])).T.astype(np.int64)

        return {
            "adj_train": adjacency,
            "adj_norm_t": adj_norm_t,
            "adj_label_t": adj_label_t,
            "features_t": features_t,
            "labels": labels,
            "norm": norm,
            "weight_tensor": weight_tensor,
            "num_nodes": graph.number_of_nodes(),
            "pos_edges_upper": pos_edges_upper,
            "neg_edges_upper": neg_edges_upper,
        }

    def _compute_vgae_loss(
        self,
        *,
        model: _VGAEModel,
        graph_pack: Dict[str, Any],
        beta: float,
    ) -> torch.Tensor:
        """
        Compute beta-VGAE objective (weighted BCE - beta*KL) for one graph pack.

        Args:
            model: VGAE model instance.
            graph_pack: One graph tensor pack.
            beta: KL multiplier.

        Returns:
            Scalar loss tensor.
        """
        adjacency_pred = model(graph_pack["features_t"], graph_pack["adj_norm_t"])
        bce = graph_pack["norm"] * F.binary_cross_entropy(
            adjacency_pred.view(-1),
            graph_pack["adj_label_t"].view(-1),
            weight=graph_pack["weight_tensor"],
        )
        kl = 0.5 / adjacency_pred.size(0) * (
            1 + 2 * model.logstd - model.mean**2 - torch.exp(model.logstd) ** 2
        ).sum(1).mean()
        ranking_loss = self._compute_ranking_loss(adjacency_pred, graph_pack)
        return bce - beta * kl + self.ranking_loss_weight * ranking_loss

    def _compute_ranking_loss(
        self,
        adjacency_pred: torch.Tensor,
        graph_pack: Dict[str, Any],
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Compute sampled pairwise ranking loss: positives should score above negatives.

        Args:
            adjacency_pred: Predicted adjacency probability matrix.
            graph_pack: Graph pack containing cached positive/negative edge sets.
            deterministic: If True, use deterministic sampling for reproducible evaluation.

        Returns:
            Scalar ranking loss tensor.
        """
        pos_edges = graph_pack["pos_edges_upper"]
        neg_edges = graph_pack["neg_edges_upper"]
        if pos_edges.shape[0] == 0 or neg_edges.shape[0] == 0:
            return adjacency_pred.new_tensor(0.0)

        n_pos = pos_edges.shape[0]
        n_neg = neg_edges.shape[0]
        n_pairs = min(n_pos, n_neg, max(1, self.ranking_num_negative))

        if deterministic:
            rng = np.random.default_rng(self.random_state + int(adjacency_pred.shape[0]))
            pos_idx = rng.choice(n_pos, size=n_pairs, replace=n_pos < n_pairs)
            neg_idx = rng.choice(n_neg, size=n_pairs, replace=n_neg < n_pairs)
        else:
            pos_idx = np.random.choice(n_pos, size=n_pairs, replace=n_pos < n_pairs)
            neg_idx = np.random.choice(n_neg, size=n_pairs, replace=n_neg < n_pairs)
        pos_pairs = pos_edges[pos_idx]
        neg_pairs = neg_edges[neg_idx]

        pos_scores = adjacency_pred[pos_pairs[:, 0], pos_pairs[:, 1]]
        neg_scores = adjacency_pred[neg_pairs[:, 0], neg_pairs[:, 1]]
        # Logistic pairwise ranking surrogate.
        return F.softplus(-(pos_scores - neg_scores)).mean()

    def _compute_dataset_metrics(
        self,
        *,
        model: _VGAEModel,
        graph_packs: Sequence[Dict[str, Any]],
        beta: float,
    ) -> Dict[str, Optional[float]]:
        """
        Compute averaged validation metrics over a list of graph packs.

        Args:
            model: VGAE model.
            graph_packs: Graph tensor packs.
            beta: KL multiplier.

        Returns:
            Dict with keys: `objective`, `bce`, `ap`.
        """
        if len(graph_packs) == 0:
            return {"objective": None, "bce": None, "ap": None}
        model.eval()
        objectives: List[float] = []
        bces: List[float] = []
        aps: List[float] = []
        from sklearn.metrics import average_precision_score
        with torch.no_grad():
            for pack in graph_packs:
                pred = model(pack["features_t"], pack["adj_norm_t"])
                bce_obj = pack["norm"] * F.binary_cross_entropy(
                    pred.view(-1),
                    pack["adj_label_t"].view(-1),
                    weight=pack["weight_tensor"],
                )
                kl_obj = 0.5 / pred.size(0) * (
                    1 + 2 * model.logstd - model.mean**2 - torch.exp(model.logstd) ** 2
                ).sum(1).mean()
                ranking_obj = self._compute_ranking_loss(pred, pack, deterministic=True)
                objective = bce_obj - beta * kl_obj + self.ranking_loss_weight * ranking_obj
                objectives.append(float(objective.item()))

                n = pred.shape[0]
                triu = np.triu_indices(n, k=1)
                pred_vals = pred.cpu().numpy()[triu]
                true_vals = pack["adj_train"].toarray().astype(np.float32)[triu]
                bce = float(F.binary_cross_entropy(torch.FloatTensor(pred_vals), torch.FloatTensor(true_vals)).item())
                bces.append(bce)
                # AP undefined if only one class exists.
                if np.unique(true_vals).shape[0] > 1:
                    aps.append(float(average_precision_score(true_vals, pred_vals)))
        return {
            "objective": float(np.mean(objectives)) if objectives else None,
            "bce": float(np.mean(bces)) if bces else None,
            "ap": float(np.mean(aps)) if aps else None,
        }

    def _beta_for_epoch(self, epoch: int) -> float:
        """
        Compute beta-VAE KL multiplier for a training epoch.

        Args:
            epoch: Zero-based epoch index.

        Returns:
            Beta value for this epoch.
        """
        if self.beta_warmup_epochs <= 0:
            return self.beta_end
        progress = min(1.0, float(epoch + 1) / float(self.beta_warmup_epochs))
        return self.beta_start + (self.beta_end - self.beta_start) * progress

    def _extract_latent_distribution(
        self,
        *,
        model: _VGAEModel,
        graph_packs: Sequence[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract weighted latent summaries from trained model over graph packs.

        Graph-level latent summaries are weighted by graph size.

        Args:
            model: Trained VGAE model.
            graph_packs: Training graph packs.

        Returns:
            Tuple of (latent_mu, latent_std).
        """
        model.eval()
        mu_list: List[torch.Tensor] = []
        std_list: List[torch.Tensor] = []
        weights: List[float] = []
        with torch.no_grad():
            for pack in graph_packs:
                _ = model(pack["features_t"], pack["adj_norm_t"])
                mu_g = model.mean.mean(dim=0)
                std_g = torch.exp(model.logstd).mean(dim=0)
                w = float(pack["num_nodes"])
                mu_list.append(mu_g)
                std_list.append(std_g)
                weights.append(w)
        w_t = torch.tensor(weights, dtype=torch.float32)
        w_t = w_t / w_t.sum()
        mu_stack = torch.stack(mu_list, dim=0)
        std_stack = torch.stack(std_list, dim=0)
        latent_mu = (mu_stack * w_t.unsqueeze(1)).sum(dim=0)
        latent_std = (std_stack * w_t.unsqueeze(1)).sum(dim=0) + 1e-4
        return latent_mu, latent_std

    def _estimate_label_distribution(self, graph_packs: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate empirical node-label distribution from training graph packs.

        Args:
            graph_packs: Training graph packs.

        Returns:
            Tuple of (label_ids, label_probs).
        """
        all_labels = np.concatenate([pack["labels"] for pack in graph_packs])
        label_counts = Counter(all_labels.tolist())
        label_ids = np.array(list(label_counts.keys()), dtype=np.int64)
        label_probs = np.array(list(label_counts.values()), dtype=np.float64)
        label_probs = label_probs / label_probs.sum()
        return label_ids, label_probs

    @staticmethod
    def _estimate_size_distribution(graphs: Sequence[nx.Graph]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate empirical graph size distribution.

        Args:
            graphs: Training graphs.

        Returns:
            Tuple of (size_values, size_probs).
        """
        sizes = [int(g.number_of_nodes()) for g in graphs]
        counts = Counter(sizes)
        size_values = np.array(sorted(counts.keys()), dtype=np.int64)
        size_probs = np.array([counts[s] for s in size_values], dtype=np.float64)
        size_probs = size_probs / size_probs.sum()
        return size_values, size_probs

    def _calibrate_edge_threshold(
        self,
        *,
        model: _VGAEModel,
        graph_packs: Sequence[Dict[str, Any]],
        previous_threshold: Optional[float] = None,
    ) -> float:
        """
        Calibrate threshold to match edge density on validation graph packs.

        Args:
            model: Trained VGAE model.
            graph_packs: Validation packs (or fallback packs).
            previous_threshold: Previous threshold value for smoothing/fallback.

        Returns:
            Calibrated threshold in [0.05, 0.95], or global default when disabled.
        """
        if previous_threshold is None:
            previous_threshold = float(self.edge_threshold)
        if not self.calibrate_edge_threshold or len(graph_packs) == 0:
            return float(previous_threshold)

        model.eval()
        all_probs: List[np.ndarray] = []
        all_truth: List[np.ndarray] = []
        with torch.no_grad():
            for pack in graph_packs:
                pred = model(pack["features_t"], pack["adj_norm_t"]).cpu().numpy()
                n = pred.shape[0]
                triu = np.triu_indices(n, k=1)
                all_probs.append(pred[triu])
                truth = pack["adj_train"].toarray().astype(np.float32)
                all_truth.append(truth[triu])

        prob_vals = np.concatenate(all_probs)
        truth_vals = np.concatenate(all_truth)
        if prob_vals.size == 0:
            return float(previous_threshold)
        if prob_vals.size < self.calibration_min_sample_size:
            return float(previous_threshold if self.calibration_fallback_to_previous else self.edge_threshold)

        true_density = float(np.mean(truth_vals))
        quantile = max(0.0, min(1.0, 1.0 - true_density))
        raw_threshold = float(np.quantile(prob_vals, quantile))
        smooth = float(np.clip(self.calibration_quantile_smoothing, 0.0, 1.0))
        threshold = smooth * float(previous_threshold) + (1.0 - smooth) * raw_threshold
        return float(np.clip(threshold, 0.05, 0.95))

    @staticmethod
    def _estimate_edge_density(graph_packs: Sequence[Dict[str, Any]]) -> Optional[float]:
        """
        Estimate mean undirected edge density over graph packs.

        Args:
            graph_packs: Graph packs containing `adj_train`.

        Returns:
            Mean edge density in [0,1], or None if unavailable.
        """
        if len(graph_packs) == 0:
            return None
        densities: List[float] = []
        for pack in graph_packs:
            adj = pack["adj_train"]
            n = int(adj.shape[0])
            if n <= 1:
                continue
            n_possible = (n * (n - 1)) / 2.0
            n_edges = float(adj.sum()) / 2.0
            densities.append(n_edges / n_possible)
        if len(densities) == 0:
            return None
        return float(np.mean(densities))

    def _fit_one_type(
        self,
        *,
        graphs: Sequence[nx.Graph],
        type_key: Hashable,
    ) -> Dict[str, Any]:
        """
        Fit one VGAE model for one target type bucket.

        Args:
            graphs: Graphs of one type.
            type_key: Bucket key.

        Returns:
            Learned state dictionary for that type.
        """
        split_seed = self.random_state + (abs(hash(str(type_key))) % 100003)
        splits = self._iter_train_val_splits(graphs, seed=split_seed)

        selected_state: Optional[Dict[str, Any]] = None
        selected_score = float("inf")
        selected_ap = -float("inf")
        split_summaries: List[Dict[str, Optional[float]]] = []

        for split_idx, (train_graphs, val_graphs) in enumerate(splits):
            scaler = self._fit_feature_scaler(train_graphs, n_label_features=self._label_feature_dim)
            train_packs = [
                self._make_graph_pack(g, scaler=scaler, n_label_features=self._label_feature_dim)
                for g in train_graphs
            ]
            val_packs = [
                self._make_graph_pack(g, scaler=scaler, n_label_features=self._label_feature_dim)
                for g in val_graphs
            ]

            input_dim = int(train_packs[0]["features_t"].shape[1])
            model = _VGAEModel(
                input_dim=input_dim,
                hidden1_dim=self.hidden1_dim,
                hidden2_dim=self.hidden2_dim,
                dropout_rate=self.dropout_rate,
                encoder_num_layers=self.encoder_num_layers,
                use_residual=self.use_residual,
            )
            optimizer = Adam(model.parameters(), lr=self.learning_rate)

            best_val_bce = float("inf")
            best_val_ap = -float("inf")
            best_state_dict = None
            best_epoch = 0
            patience_counter = 0

            for epoch in range(self.num_epochs):
                beta = self._beta_for_epoch(epoch)
                model.train()
                train_objectives: List[float] = []
                for pack in train_packs:
                    optimizer.zero_grad()
                    objective = self._compute_vgae_loss(model=model, graph_pack=pack, beta=beta)
                    objective.backward()
                    optimizer.step()
                    train_objectives.append(float(objective.item()))
                train_objective = float(np.mean(train_objectives))

                val_metrics = self._compute_dataset_metrics(model=model, graph_packs=val_packs, beta=beta)
                val_objective = val_metrics["objective"]
                val_bce = val_metrics["bce"]
                val_ap = val_metrics["ap"]

                if val_bce is not None:
                    improved_bce = val_bce < (best_val_bce - self.early_stopping_min_delta)
                    improved_ap_tie = abs(val_bce - best_val_bce) <= self.early_stopping_min_delta and (
                        val_ap is not None and val_ap > best_val_ap
                    )
                    if improved_bce or improved_ap_tie:
                        best_val_bce = val_bce
                        best_val_ap = val_ap if val_ap is not None else best_val_ap
                        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        best_epoch = epoch + 1
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if self.early_stopping_patience > 0 and patience_counter >= self.early_stopping_patience:
                            if self.verbose:
                                print(
                                    f"[VGAE][type={type_key}][split={split_idx}] "
                                    f"early_stopped epoch={epoch + 1:03d} "
                                    f"best_epoch={best_epoch:03d}"
                                )
                            break

                if self.verbose:
                    patience_max = max(0, self.early_stopping_patience)
                    if val_bce is not None and patience_max > 0:
                        patience_str = f"{patience_counter}/{patience_max}"
                    else:
                        patience_str = f"0/{patience_max}" if patience_max > 0 else "disabled"
                    msg = (
                        f"[VGAE][type={type_key}][split={split_idx}] "
                        f"epoch={epoch + 1:03d}/{self.num_epochs:03d} "
                        f"beta={beta:.4f} train_obj={train_objective:.6f} "
                        f"patience={patience_str}"
                    )
                    if val_objective is not None:
                        msg += f" val_obj={val_objective:.6f}"
                    if val_bce is not None:
                        msg += f" val_bce={val_bce:.6f}"
                    if val_ap is not None:
                        msg += f" val_ap={val_ap:.6f}"
                    print(msg)

            if self.restore_best_checkpoint and best_state_dict is not None:
                model.load_state_dict(best_state_dict)

            latent_mu, latent_std = self._extract_latent_distribution(model=model, graph_packs=train_packs)
            label_ids, label_probs = self._estimate_label_distribution(train_packs)
            size_values, size_probs = self._estimate_size_distribution(train_graphs)
            threshold_source = val_packs if len(val_packs) > 0 else train_packs
            target_edge_density = self._estimate_edge_density(threshold_source)
            # Use a stable baseline threshold per split to avoid split-order dependence.
            previous_threshold = self.edge_threshold
            calibrated_threshold = self._calibrate_edge_threshold(
                model=model,
                graph_packs=threshold_source,
                previous_threshold=float(previous_threshold),
            )

            final_val_metrics = self._compute_dataset_metrics(model=model, graph_packs=val_packs, beta=self.beta_end)
            final_train_metrics = self._compute_dataset_metrics(model=model, graph_packs=train_packs, beta=self.beta_end)
            # Rank split by the best tracked validation reconstruction metric (checkpoint criterion).
            split_bce = best_val_bce if np.isfinite(best_val_bce) else final_val_metrics["bce"]
            split_ap = best_val_ap if np.isfinite(best_val_ap) else final_val_metrics["ap"]
            split_bce_finite = split_bce is not None and np.isfinite(float(split_bce))
            split_ap_finite = split_ap is not None and np.isfinite(float(split_ap))

            split_state = {
                "type_key": type_key,
                "model": model,
                "feature_scaler": scaler,
                "latent_mu": latent_mu,
                "latent_std": latent_std,
                "label_ids": label_ids,
                "label_probs": label_probs,
                "size_values": size_values,
                "size_probs": size_probs,
                "edge_threshold": calibrated_threshold,
                "target_edge_density": target_edge_density,
                "default_num_nodes": int(round(float(np.mean(size_values)))),
                "train_graphs": list(train_graphs),
                "val_graphs": list(val_graphs),
            }

            split_summaries.append(
                {
                    "best_epoch": int(best_epoch) if best_epoch > 0 else None,
                    "val_objective": final_val_metrics["objective"],
                    "train_objective": final_train_metrics["objective"],
                    "train_bce": final_train_metrics["bce"],
                    "train_ap": final_train_metrics["ap"],
                    "val_bce": float(split_bce) if split_bce_finite else None,
                    "val_ap": float(split_ap) if split_ap_finite else None,
                }
            )

            # If no validation metric exists, fall back to train BCE for split selection.
            if split_bce_finite:
                comp_bce = float(split_bce)
            else:
                comp_bce = (
                    float(final_train_metrics["bce"])
                    if final_train_metrics["bce"] is not None and np.isfinite(final_train_metrics["bce"])
                    else float("inf")
                )
            comp_ap = (
                float(split_ap)
                if split_ap_finite
                else (
                    float(final_train_metrics["ap"])
                    if final_train_metrics["ap"] is not None and np.isfinite(final_train_metrics["ap"])
                    else -float("inf")
                )
            )
            if selected_state is None or (comp_bce < selected_score - 1e-12) or (
                abs(comp_bce - selected_score) <= 1e-12 and comp_ap > selected_ap
            ):
                selected_score = comp_bce
                selected_ap = comp_ap
                selected_state = split_state

        assert selected_state is not None
        selected_state["split_metrics"] = split_summaries
        # Aggregate summary across splits.
        bces = [m["val_bce"] for m in split_summaries if m["val_bce"] is not None]
        aps = [m["val_ap"] for m in split_summaries if m["val_ap"] is not None]
        objs = [m["val_objective"] for m in split_summaries if m["val_objective"] is not None]
        selected_state["split_metrics_agg"] = {
            "val_objective_mean": float(np.mean(objs)) if objs else None,
            "val_objective_std": float(np.std(objs)) if objs else None,
            "val_bce_mean": float(np.mean(bces)) if bces else None,
            "val_bce_std": float(np.std(bces)) if bces else None,
            "val_ap_mean": float(np.mean(aps)) if aps else None,
            "val_ap_std": float(np.std(aps)) if aps else None,
        }
        return selected_state

    def fit(
        self,
        graphs: Sequence[nx.Graph],
        targets: Optional[Sequence[Hashable]] = None,
    ) -> "VGAEGraphGenerator":
        """
        Fit on a collection of NetworkX graphs.

        Args:
            graphs: Graph list for training.
            targets: Optional labels for per-type model fitting.

        Returns:
            Self.
        """
        if len(graphs) == 0:
            raise ValueError("graphs must not be empty.")
        if targets is not None and len(targets) != len(graphs):
            raise ValueError("targets must have the same length as graphs.")

        self._set_random_seeds(self.random_state)
        self._label_feature_dim = self._infer_label_feature_dim(graphs)

        if targets is None:
            self._states = {"default": self._fit_one_type(graphs=graphs, type_key="default")}
        else:
            buckets = self._bucket_graphs_by_target(graphs, targets)
            self._states = {}
            for type_key, type_graphs in buckets.items():
                self._states[type_key] = self._fit_one_type(graphs=type_graphs, type_key=type_key)

        self._is_fitted = True
        return self

    def _resolve_state_for_generation(self, graph_type: Optional[Hashable]) -> Tuple[Hashable, Dict[str, Any]]:
        """
        Resolve which fitted state to use for generation.

        Args:
            graph_type: Optional requested type key.

        Returns:
            Tuple of (resolved_graph_type, state_dict).
        """
        if len(self._states) == 1 and graph_type is None:
            graph_type = next(iter(self._states.keys()))
        if graph_type is None:
            raise ValueError("graph_type is required when multiple type models are fitted.")
        if graph_type not in self._states:
            raise KeyError(f"Unknown graph_type '{graph_type}'. Known keys: {list(self._states.keys())}")
        return graph_type, self._states[graph_type]

    def _sample_adjacency_binary(
        self,
        *,
        mu: torch.Tensor,
        std: torch.Tensor,
        num_nodes: int,
        edge_threshold: float,
        target_edge_density: Optional[float],
        sample_bernoulli: bool,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Sample a symmetric binary adjacency matrix from latent Gaussian.

        Args:
            mu: Latent mean vector.
            std: Latent std vector.
            num_nodes: Number of nodes.
            edge_threshold: Threshold for deterministic edge binarization.
            target_edge_density: Optional target edge density for dynamic threshold matching.
            sample_bernoulli: If True, sample edges from Bernoulli probabilities.
            rng: Random number generator.

        Returns:
            Binary adjacency matrix as ndarray.
        """
        eps = torch.randn(num_nodes, mu.numel())
        z = mu.unsqueeze(0) + eps * std.unsqueeze(0)
        adjacency_prob = torch.sigmoid(z @ z.t()).cpu().numpy()
        upper = np.triu(adjacency_prob, k=1)
        triu_idx = np.triu_indices(num_nodes, k=1)
        if sample_bernoulli:
            adjacency_bin = rng.binomial(1, upper).astype(np.int64)
        else:
            threshold_eff = float(edge_threshold)
            if self.dynamic_density_matching and target_edge_density is not None:
                dens = float(np.clip(target_edge_density, 1e-6, 1.0 - 1e-6))
                q = float(np.clip(1.0 - dens, 0.0, 1.0))
                dynamic_thr = float(np.quantile(upper[triu_idx], q))
                blend = float(np.clip(self.density_match_blend, 0.0, 1.0))
                threshold_eff = (1.0 - blend) * threshold_eff + blend * dynamic_thr
            adjacency_bin = (upper > threshold_eff).astype(np.int64)
        adjacency_bin = adjacency_bin + adjacency_bin.T

        if self.enforce_connected and num_nodes > 1:
            adjacency_bin = self._enforce_connectivity(
                adjacency_bin=adjacency_bin,
                upper_prob=upper,
                initial_threshold=edge_threshold,
                triu_idx=triu_idx,
            )
        return adjacency_bin

    def _enforce_connectivity(
        self,
        *,
        adjacency_bin: np.ndarray,
        upper_prob: np.ndarray,
        initial_threshold: float,
        triu_idx: Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """
        Lower threshold progressively to obtain connected graph, then fallback by linking components.

        Args:
            adjacency_bin: Current binary adjacency matrix.
            upper_prob: Upper-triangular edge probabilities.
            initial_threshold: Starting threshold.
            triu_idx: Upper-triangular index tuple.

        Returns:
            Connected binary adjacency matrix.
        """
        graph = nx.from_numpy_array(adjacency_bin)
        if graph.number_of_nodes() <= 1 or nx.is_connected(graph):
            return adjacency_bin

        threshold = float(initial_threshold)
        step = max(1e-6, float(self.connectivity_threshold_step))
        min_thr = float(np.clip(self.connectivity_min_threshold, 0.0, 1.0))
        max_iters = max(1, int(self.connectivity_max_iters))

        for _ in range(max_iters):
            threshold = max(min_thr, threshold - step)
            adjacency_candidate = (upper_prob > threshold).astype(np.int64)
            adjacency_candidate = adjacency_candidate + adjacency_candidate.T
            graph = nx.from_numpy_array(adjacency_candidate)
            if nx.is_connected(graph):
                return adjacency_candidate
            if threshold <= min_thr:
                adjacency_bin = adjacency_candidate
                break
            adjacency_bin = adjacency_candidate

        # Final fallback: connect components using highest-probability inter-component edges.
        graph = nx.from_numpy_array(adjacency_bin)
        components = [list(c) for c in nx.connected_components(graph)]
        while len(components) > 1:
            base = components[0]
            best_edge = None
            best_prob = -1.0
            for c_idx in range(1, len(components)):
                comp = components[c_idx]
                for u in base:
                    for v in comp:
                        i, j = (u, v) if u < v else (v, u)
                        p = float(upper_prob[i, j])
                        if p > best_prob:
                            best_prob = p
                            best_edge = (u, v)
            if best_edge is None:
                break
            u, v = best_edge
            adjacency_bin[u, v] = 1
            adjacency_bin[v, u] = 1
            graph = nx.from_numpy_array(adjacency_bin)
            components = [list(c) for c in nx.connected_components(graph)]

        return adjacency_bin

    @staticmethod
    def _assign_sampled_labels(
        graph: nx.Graph,
        *,
        label_ids: np.ndarray,
        label_probs: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """
        Sample and assign node labels to a graph in-place.

        Args:
            graph: Graph to label.
            label_ids: Possible label ids.
            label_probs: Sampling probabilities.
            rng: NumPy generator.

        Returns:
            None.
        """
        sampled_labels = rng.choice(label_ids, size=graph.number_of_nodes(), p=label_probs)
        for node in graph.nodes():
            graph.nodes[node]["label"] = int(sampled_labels[node])

    def generate(
        self,
        *,
        num_graphs: int,
        graph_type: Optional[Hashable] = None,
        num_nodes: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[nx.Graph]:
        """
        Generate sampled graphs from fitted model(s).

        Args:
            num_graphs: Number of graphs to generate.
            graph_type: Type key to sample from.
            num_nodes: Nodes per generated graph. If omitted, sampled from empirical size distribution.
            seed: Optional override for random seed.

        Returns:
            List of generated NetworkX graphs.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit(...) before generate(...).")

        _resolved_graph_type, state = self._resolve_state_for_generation(graph_type)
        mu = state["latent_mu"]
        std = state["latent_std"]
        label_ids = state["label_ids"]
        label_probs = state["label_probs"]
        edge_threshold = float(state.get("edge_threshold", self.edge_threshold))
        target_edge_density = state.get("target_edge_density")
        size_values = state.get("size_values")
        size_probs = state.get("size_probs")

        if seed is None:
            seed = self.random_state
        self._set_random_seeds(seed)
        rng = np.random.default_rng(seed)

        out: List[nx.Graph] = []
        for _ in range(num_graphs):
            if num_nodes is None:
                if size_values is not None and size_probs is not None:
                    sampled_nodes = int(rng.choice(size_values, p=size_probs))
                else:
                    sampled_nodes = int(state["default_num_nodes"])
            else:
                sampled_nodes = int(num_nodes)

            if sampled_nodes <= 0:
                raise ValueError("num_nodes must be positive.")

            adjacency_bin = self._sample_adjacency_binary(
                mu=mu,
                std=std,
                num_nodes=sampled_nodes,
                edge_threshold=edge_threshold,
                target_edge_density=target_edge_density,
                sample_bernoulli=self.sample_edges_bernoulli,
                rng=rng,
            )
            graph = nx.from_numpy_array(adjacency_bin)
            self._assign_sampled_labels(
                graph,
                label_ids=label_ids,
                label_probs=label_probs,
                rng=rng,
            )
            out.append(graph)
        return out
