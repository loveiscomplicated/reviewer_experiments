from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    global_add_pool,
    global_mean_pool,
)

try:
    from reviewer_experiments.tensor_models import NodeEncoder
except ModuleNotFoundError:
    from tensor_models import NodeEncoder  # type: ignore


def edge_tensors_from_adjacency(adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    edge_index = (adj > 0).nonzero(as_tuple=False).t().contiguous()
    edge_weight = adj[edge_index[0], edge_index[1]].float()
    return edge_index.long(), edge_weight


def repeat_edge_tensors(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    batch_size: int,
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    offsets = (
        torch.arange(batch_size, device=edge_index.device, dtype=edge_index.dtype)
        * num_nodes
    )
    batched_edge_index = edge_index.unsqueeze(0) + offsets[:, None, None]
    batched_edge_index = batched_edge_index.permute(1, 0, 2).reshape(2, -1).contiguous()
    if edge_weight is None:
        return batched_edge_index, None
    return batched_edge_index, edge_weight.repeat(batch_size)


def batch_vector(batch_size: int, num_nodes: int, device: torch.device) -> torch.Tensor:
    return torch.arange(batch_size, device=device).repeat_interleave(num_nodes)


class PYGGraphMixin:
    encoder: NodeEncoder

    def encode_flat(self, x_id: torch.Tensor, x_missing: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x_id, x_missing)
        batch_size, num_nodes, feat_dim = x.shape
        return x.reshape(batch_size * num_nodes, feat_dim)

    def cached_graph_tensors(
        self,
        adj: torch.Tensor,
        repeat_count: int,
        num_nodes: int,
        device: torch.device,
        use_edge_weight: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        cache: Dict[
            Tuple[int, str, int, int, bool],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
        ] = getattr(self, "_graph_tensor_cache", {})
        key = (id(adj), str(device), repeat_count, num_nodes, use_edge_weight)
        cached = cache.get(key)
        if cached is None:
            adj_on_device = adj if adj.device == device else adj.to(device)
            base_edge_index, base_edge_weight = edge_tensors_from_adjacency(
                adj_on_device
            )
            edge_index, edge_weight = repeat_edge_tensors(
                base_edge_index,
                base_edge_weight if use_edge_weight else None,
                repeat_count,
                num_nodes,
            )
            batch = batch_vector(repeat_count, num_nodes, device)
            cached = (edge_index, batch, edge_weight)
            cache[key] = cached
            setattr(self, "_graph_tensor_cache", cache)
        return cached

    def graph_inputs(
        self,
        x_id: torch.Tensor,
        x_missing: torch.Tensor,
        adj: torch.Tensor,
        use_edge_weight: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x_flat = self.encode_flat(x_id, x_missing)
        batch_size, num_nodes = x_id.shape
        edge_index, batch, edge_weight = self.cached_graph_tensors(
            adj,
            batch_size,
            num_nodes,
            x_flat.device,
            use_edge_weight,
        )
        return x_flat, edge_index, batch, edge_weight


class PYGGCNClassifier(nn.Module, PYGGraphMixin):
    def __init__(
        self,
        cat_dims: Sequence[int],
        hidden_dim: int = 64,
        num_classes: int = 2,
        num_layers: int = 3,
        dropout: float = 0.5,
        proj_dim: int = 7,
    ):
        super().__init__()
        self.encoder = NodeEncoder(cat_dims, proj_dim=proj_dim)
        dims = [self.encoder.out_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self, x_id: torch.Tensor, x_missing: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        x, edge_index, batch, edge_weight = self.graph_inputs(
            x_id, x_missing, adj, use_edge_weight=True
        )
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = self.dropout(x)
        return self.classifier(global_mean_pool(x, batch))


class PYGGINClassifier(nn.Module, PYGGraphMixin):
    def __init__(
        self,
        cat_dims: Sequence[int],
        hidden_dim: int = 64,
        num_classes: int = 2,
        num_layers: int = 5,
        dropout: float = 0.5,
        proj_dim: int = 7,
    ):
        super().__init__()
        self.encoder = NodeEncoder(cat_dims, proj_dim=proj_dim)
        self.convs = nn.ModuleList()
        self.convs.append(
            GINConv(
                nn.Sequential(
                    nn.Linear(self.encoder.out_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
        )
        for _ in range(1, num_layers):
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                )
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, x_id: torch.Tensor, x_missing: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        x, edge_index, batch, _ = self.graph_inputs(
            x_id, x_missing, adj, use_edge_weight=False
        )
        layer_outputs = []
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            layer_outputs.append(x)
        graph_repr = global_add_pool(torch.cat(layer_outputs, dim=-1), batch)
        graph_repr = self.dropout(graph_repr)
        return self.classifier(graph_repr)


class PYGGATClassifier(nn.Module, PYGGraphMixin):
    def __init__(
        self,
        cat_dims: Sequence[int],
        hidden_dim: int = 64,
        num_classes: int = 2,
        heads: int = 4,
        dropout: float = 0.5,
        proj_dim: int = 7,
    ):
        super().__init__()
        self.encoder = NodeEncoder(cat_dims, proj_dim=proj_dim)
        self.gat1 = GATConv(
            self.encoder.out_dim, hidden_dim, heads=heads, dropout=dropout
        )
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self, x_id: torch.Tensor, x_missing: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        x, edge_index, batch, _ = self.graph_inputs(
            x_id, x_missing, adj, use_edge_weight=False
        )
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat2(x, edge_index))
        return self.classifier(global_mean_pool(x, batch))


class PYGTGCNClassifier(nn.Module, PYGGraphMixin):
    def __init__(
        self,
        cat_dims: Sequence[int],
        gnn_hidden_dim: int = 64,
        rnn_hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.5,
        proj_dim: int = 7,
    ):
        super().__init__()
        self.encoder = NodeEncoder(cat_dims, proj_dim=proj_dim)
        self.gcn1 = GCNConv(self.encoder.out_dim, gnn_hidden_dim)
        self.gcn2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim)
        self.rnn = nn.GRU(
            gnn_hidden_dim, rnn_hidden_dim, num_layers=1, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(rnn_hidden_dim, num_classes)

    def forward(
        self, x_id: torch.Tensor, x_missing: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        if x_id.dim() != 3:
            raise ValueError(
                f"T-GCN x_id must be [batch, time, nodes], got {tuple(x_id.shape)}"
            )
        if x_missing.dim() != 4:
            raise ValueError(
                f"T-GCN x_missing must be [batch, time, nodes, 1], got {tuple(x_missing.shape)}"
            )
        batch_size, timesteps, num_nodes = x_id.shape
        flat_id = x_id.reshape(batch_size * timesteps, num_nodes)
        flat_missing = x_missing.reshape(batch_size * timesteps, num_nodes, 1)
        x = self.encoder(flat_id, flat_missing)
        x = x.reshape(batch_size * timesteps * num_nodes, -1)

        edge_index, batch, edge_weight = self.cached_graph_tensors(
            adj,
            batch_size * timesteps,
            num_nodes,
            x.device,
            use_edge_weight=True,
        )

        x = F.relu(self.gcn1(x, edge_index, edge_weight=edge_weight))
        x = self.dropout(x)
        x = F.relu(self.gcn2(x, edge_index, edge_weight=edge_weight))
        graph_embeddings = global_mean_pool(x, batch).view(batch_size, timesteps, -1)
        _, hidden = self.rnn(graph_embeddings)
        return self.classifier(self.dropout(hidden[-1]))


def build_pyg_model(
    model_name: str,
    cat_dims: Sequence[int],
    hidden_dim: int = 64,
    num_classes: int = 2,
    dropout: float = 0.5,
    proj_dim: int = 7,
) -> nn.Module:
    if model_name == "gcn":
        return PYGGCNClassifier(
            cat_dims=cat_dims,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            proj_dim=proj_dim,
        )
    if model_name == "gin":
        return PYGGINClassifier(
            cat_dims=cat_dims,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            proj_dim=proj_dim,
        )
    if model_name == "gat":
        return PYGGATClassifier(
            cat_dims=cat_dims,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            proj_dim=proj_dim,
        )
    if model_name == "tgcn":
        return PYGTGCNClassifier(
            cat_dims=cat_dims,
            gnn_hidden_dim=hidden_dim,
            rnn_hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            proj_dim=proj_dim,
        )
    raise ValueError(f"Unknown model: {model_name}")
