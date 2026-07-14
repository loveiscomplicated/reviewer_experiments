from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeEncoder(nn.Module):
    """Variable-wise categorical embeddings projected to a shared 7D space."""

    def __init__(self, cat_dims: Sequence[int], proj_dim: int = 7):
        super().__init__()
        self.cat_dims = list(cat_dims)
        self.proj_dim = proj_dim
        self.embeddings = nn.ModuleList()
        self.projections = nn.ModuleList()

        for num_categories in self.cat_dims:
            emb_dim = int(math.ceil(num_categories**0.5))
            self.embeddings.append(nn.Embedding(num_categories, emb_dim))
            self.projections.append(nn.Linear(emb_dim, proj_dim))

    @property
    def out_dim(self) -> int:
        return self.proj_dim + 1

    def forward(self, x_id: torch.Tensor, x_missing: torch.Tensor) -> torch.Tensor:
        if x_id.dim() != 2:
            raise ValueError(f"x_id must be [batch, nodes], got {tuple(x_id.shape)}")
        if x_missing.dim() != 3:
            raise ValueError(f"x_missing must be [batch, nodes, 1], got {tuple(x_missing.shape)}")
        if x_id.shape[1] != len(self.embeddings):
            raise ValueError(f"Expected {len(self.embeddings)} nodes, got {x_id.shape[1]}")

        encoded = []
        for node_idx, (embedding, projection) in enumerate(zip(self.embeddings, self.projections)):
            out = projection(embedding(x_id[:, node_idx]))
            encoded.append(out)
        stacked = torch.stack(encoded, dim=1)
        return torch.cat([stacked, x_missing.float()], dim=-1)


def normalized_adjacency(adj: torch.Tensor) -> torch.Tensor:
    adj = adj.float()
    eye = torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    adj_hat = adj + eye
    degree = adj_hat.sum(dim=1).clamp_min(1e-12)
    d_inv_sqrt = degree.pow(-0.5)
    return d_inv_sqrt[:, None] * adj_hat * d_inv_sqrt[None, :]


class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("ij,bjf->bif", adj_norm, x)
        x = self.linear(x)
        x = F.relu(x)
        return self.dropout(x)


class TensorGCNClassifier(nn.Module):
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
        self.layers = nn.ModuleList(
            DenseGCNLayer(dims[i], dims[i + 1], dropout=dropout) for i in range(num_layers)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_id: torch.Tensor, x_missing: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x_id, x_missing)
        adj_norm = normalized_adjacency(adj.to(x.device))
        for layer in self.layers:
            x = layer(x, adj_norm)
        graph_repr = x.mean(dim=1)
        return self.classifier(graph_repr)


class DenseGINLayer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        aggregated = torch.einsum("ij,bjf->bif", adj, x)
        out = (1.0 + self.eps) * x + aggregated
        return F.relu(self.mlp(out))


class TensorGINClassifier(nn.Module):
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
        self.layers = nn.ModuleList()
        self.layers.append(DenseGINLayer(self.encoder.out_dim, hidden_dim))
        for _ in range(1, num_layers):
            self.layers.append(DenseGINLayer(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x_id: torch.Tensor, x_missing: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x_id, x_missing)
        adj = adj.to(x.device).float()
        outputs = []
        for layer in self.layers:
            x = layer(x, adj)
            outputs.append(x)
        graph_repr = torch.cat(outputs, dim=-1).sum(dim=1)
        graph_repr = self.dropout(graph_repr)
        return self.classifier(graph_repr)


class DenseGATLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.5,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.linear = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        self.dropout = nn.Dropout(dropout)
        self.negative_slope = negative_slope
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        h = self.linear(x).view(batch_size, num_nodes, self.heads, self.out_dim)
        h = h.permute(0, 2, 1, 3)

        src_score = (h * self.att_src[None, :, None, :]).sum(dim=-1)
        dst_score = (h * self.att_dst[None, :, None, :]).sum(dim=-1)
        logits = src_score[:, :, :, None] + dst_score[:, :, None, :]
        logits = F.leaky_relu(logits, negative_slope=self.negative_slope)

        mask = (adj.to(x.device) > 0)
        eye = torch.eye(num_nodes, dtype=torch.bool, device=x.device)
        mask = mask | eye
        logits = logits.masked_fill(~mask[None, None, :, :], torch.finfo(logits.dtype).min)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhij,bhjf->bhif", attn, h)

        if self.concat:
            out = out.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)
        return out + self.bias


class TensorGATClassifier(nn.Module):
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
        self.gat1 = DenseGATLayer(self.encoder.out_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.gat2 = DenseGATLayer(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_id: torch.Tensor, x_missing: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x_id, x_missing)
        x = F.elu(self.gat1(x, adj))
        x = self.dropout(x)
        x = F.elu(self.gat2(x, adj))
        graph_repr = x.mean(dim=1)
        return self.classifier(graph_repr)


class TensorTGCNClassifier(nn.Module):
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
        self.gcn1 = DenseGCNLayer(self.encoder.out_dim, gnn_hidden_dim, dropout=dropout)
        self.gcn2 = DenseGCNLayer(gnn_hidden_dim, gnn_hidden_dim, dropout=dropout)
        self.rnn = nn.GRU(gnn_hidden_dim, rnn_hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(rnn_hidden_dim, num_classes)

    def forward(self, x_id: torch.Tensor, x_missing: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if x_id.dim() != 3:
            raise ValueError(f"T-GCN x_id must be [batch, time, nodes], got {tuple(x_id.shape)}")
        if x_missing.dim() != 4:
            raise ValueError(
                f"T-GCN x_missing must be [batch, time, nodes, 1], got {tuple(x_missing.shape)}"
            )

        batch_size, timesteps, num_nodes = x_id.shape
        flat_id = x_id.reshape(batch_size * timesteps, num_nodes)
        flat_missing = x_missing.reshape(batch_size * timesteps, num_nodes, 1)
        x = self.encoder(flat_id, flat_missing)

        adj_norm = normalized_adjacency(adj.to(x.device))
        x = self.gcn1(x, adj_norm)
        x = self.gcn2(x, adj_norm)
        graph_embeddings = x.mean(dim=1).view(batch_size, timesteps, -1)
        _, hidden = self.rnn(graph_embeddings)
        out = self.dropout(hidden[-1])
        return self.classifier(out)


def build_model(
    model_name: str,
    cat_dims: Sequence[int],
    hidden_dim: int = 64,
    num_classes: int = 2,
    dropout: float = 0.5,
    proj_dim: int = 7,
) -> nn.Module:
    if model_name == "gcn":
        return TensorGCNClassifier(
            cat_dims=cat_dims,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            proj_dim=proj_dim,
        )
    if model_name == "gin":
        return TensorGINClassifier(
            cat_dims=cat_dims,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            proj_dim=proj_dim,
        )
    if model_name == "gat":
        return TensorGATClassifier(
            cat_dims=cat_dims,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            proj_dim=proj_dim,
        )
    if model_name == "tgcn":
        return TensorTGCNClassifier(
            cat_dims=cat_dims,
            gnn_hidden_dim=hidden_dim,
            rnn_hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            proj_dim=proj_dim,
        )
    raise ValueError(f"Unknown model: {model_name}")
