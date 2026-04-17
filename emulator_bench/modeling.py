import re
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_scatter import scatter_mean


def graph_collate_fn(batch: Sequence[Tuple]):
    graph_batch = torch_geometric.data.Batch.from_data_list([item[0] for item in batch])
    protein_tokens = torch.stack([item[1] for item in batch], dim=0)
    smiles_tokens = torch.stack([item[2] for item in batch], dim=0)
    features = torch.stack([item[3] for item in batch], dim=0)
    labels = torch.stack([item[4] for item in batch], dim=0)
    metadata = [item[5] for item in batch]
    return graph_batch, protein_tokens, smiles_tokens, features, labels, metadata


class FeedForward(nn.Module):
    def __init__(self, hidden: int, out: int, dropout: float = 0.1, residual: bool = False):
        super().__init__()
        self.residual = residual
        self.linear1 = nn.Linear(hidden, 2 * hidden)
        self.linear2 = nn.Linear(2 * hidden, out, bias=False)
        self.norm = nn.LayerNorm(out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_update = self.linear2(self.dropout(self.linear1(x)))
        if self.residual:
            return self.norm(x + self.dropout(x_update))
        return x_update


class CNN(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, kernel_size: int = 9):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.dropout1(self.bn1(self.conv1(x))))
        x = F.relu(self.dropout2(self.bn2(self.conv2(x))))
        return x + residual


class SeqEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 32, num_layers: int = 3, seq_len: int = 1000, dropout: float = 0.1, kernel_size: int = 9):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(seq_len, hidden)
        self.encoder = nn.ModuleList([CNN(hidden, dropout, kernel_size=kernel_size) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_emb(torch.arange(x.shape[1], device=x.device)) + x
        x = x.permute(0, 2, 1)
        for layer in self.encoder:
            x = layer(x)
        return x


class EdgeMLP(nn.Module):
    def __init__(self, num_hidden: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.w11 = nn.Linear(3 * num_hidden, num_hidden, bias=True)
        self.w12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()

    def forward(self, h_v: torch.Tensor, edge_index: torch.Tensor, h_e: torch.Tensor) -> torch.Tensor:
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        h_ev = torch.cat([h_v[src_idx], h_e, h_v[dst_idx]], dim=-1)
        h_message = self.w12(self.act(self.w11(h_ev)))
        return self.norm(h_e + self.dropout(h_message))


class Context(nn.Module):
    def __init__(self, num_hidden: int):
        super().__init__()
        self.v_mlp_g = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.Sigmoid(),
        )

    def forward(self, h_v: torch.Tensor, batch_id: torch.Tensor) -> torch.Tensor:
        c_v = scatter_mean(h_v, batch_id, dim=0)
        return h_v * self.v_mlp_g(c_v[batch_id])


class GATLayer(nn.Module):
    def __init__(self, num_hidden: int, dropout: float = 0.2, num_heads: int = 4):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])
        self.attention = TransformerConv(
            in_channels=num_hidden,
            out_channels=int(num_hidden / num_heads),
            heads=num_heads,
            dropout=dropout,
            edge_dim=num_hidden,
            root_weight=False,
        )
        self.positionwise = nn.Sequential(
            nn.Linear(num_hidden, num_hidden * 4),
            nn.ReLU(),
            nn.Linear(num_hidden * 4, num_hidden),
        )
        self.edge_update = EdgeMLP(num_hidden, dropout)
        self.context = Context(num_hidden)

    def forward(self, h_v: torch.Tensor, edge_index: torch.Tensor, h_e: torch.Tensor, batch_id: torch.Tensor):
        dh = self.attention(h_v, edge_index, h_e)
        h_v = self.norm[0](h_v + self.dropout(dh))
        dh = self.positionwise(h_v)
        h_v = self.norm[1](h_v + self.dropout(dh))
        h_e = self.edge_update(h_v, edge_index, h_e)
        h_v = self.context(h_v, batch_id)
        return h_v, h_e


class GraphEncoder(nn.Module):
    def __init__(self, node_in_dim: int, edge_in_dim: int, hidden_dim: int = 16, num_layers: int = 3, dropout: float = 0.5):
        super().__init__()
        self.node_project = nn.Linear(node_in_dim, 64, bias=True)
        self.edge_project = nn.Linear(edge_in_dim, 64, bias=True)
        self.bn_node = nn.BatchNorm1d(64)
        self.bn_edge = nn.BatchNorm1d(64)
        self.w_v = nn.Linear(64, hidden_dim, bias=True)
        self.w_e = nn.Linear(64, hidden_dim, bias=True)
        self.layers = nn.ModuleList([GATLayer(num_hidden=hidden_dim, dropout=dropout) for _ in range(num_layers)])

    def forward(self, graph) -> torch.Tensor:
        h_v, edge_index, h_e, batch_id = graph.x, graph.edge_index, graph.edge_attr, graph.batch
        h_v = self.w_v(self.bn_node(self.node_project(h_v)))
        h_e = self.w_e(self.bn_edge(self.edge_project(h_e)))
        for layer in self.layers:
            h_v, h_e = layer(h_v, edge_index, h_e, batch_id)
        return global_mean_pool(x=h_v, batch=batch_id).unsqueeze(1)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 32, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(input_dim, max(1, input_dim // 2))
        self.layer_norm1 = nn.LayerNorm(max(1, input_dim // 2))
        self.res_fc1 = nn.Linear(input_dim, max(1, input_dim // 2))
        self.fc2 = nn.Linear(max(1, input_dim // 2), max(1, input_dim // 4))
        self.layer_norm2 = nn.LayerNorm(max(1, input_dim // 4))
        self.res_fc2 = nn.Linear(max(1, input_dim // 2), max(1, input_dim // 4))
        self.fc3 = nn.Linear(max(1, input_dim // 4), output_dim)
        self.layer_norm3 = nn.LayerNorm(output_dim)
        self.res_fc3 = nn.Linear(max(1, input_dim // 4), output_dim)
        self.res_final = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_to_final = F.dropout(self.res_final(x), p=self.dropout, training=self.training)
        residual = x
        x = F.dropout(self.relu(self.layer_norm1(self.fc1(x))), p=self.dropout, training=self.training)
        x = x + F.dropout(self.res_fc1(residual), p=self.dropout, training=self.training)
        residual = x
        x = F.dropout(self.relu(self.layer_norm2(self.fc2(x))), p=self.dropout, training=self.training)
        x = x + F.dropout(self.res_fc2(residual), p=self.dropout, training=self.training)
        residual = x
        x = F.dropout(self.relu(self.layer_norm3(self.fc3(x))), p=self.dropout, training=self.training)
        x = x + F.dropout(self.res_fc3(residual), p=self.dropout, training=self.training)
        return x + residual_to_final


class MetaDecoder(nn.Module):
    def __init__(
        self,
        seq_vocab_size: int,
        smi_vocab_size: int,
        feature_dim_list: List[int],
        hidden: int = 32,
        num_layers: int = 3,
        protein_len: int = 2590,
        smi_len: int = 555,
        dropout: float = 0.1,
        kernel_size: int = 9,
    ):
        super().__init__()
        self.gnn_encoder = GraphEncoder(node_in_dim=42, edge_in_dim=92, hidden_dim=hidden, num_layers=num_layers, dropout=dropout)
        self.seq_encoder = SeqEncoder(seq_vocab_size, hidden=hidden, num_layers=num_layers, seq_len=protein_len, dropout=dropout, kernel_size=kernel_size)
        self.smi_encoder = SeqEncoder(smi_vocab_size, hidden=hidden, num_layers=num_layers, seq_len=smi_len, dropout=dropout, kernel_size=kernel_size)
        self.feature_dim_list = feature_dim_list
        self.feature_process = nn.ModuleList([ResidualMLP(fea_dim, hidden, dropout) for fea_dim in feature_dim_list])

        fuse_fea_dim = len(feature_dim_list) * hidden + hidden * 3
        self.fc1 = nn.Linear(fuse_fea_dim, hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, max(1, hidden // 2))
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(max(1, hidden // 2), 1)
        self.latent = None

    def forward(self, graph, protein, smi, feature):
        protein_fea = self.seq_encoder(protein)
        protein_fea = F.normalize(protein_fea, p=2, dim=-1).mean(dim=2)
        smi_fea = self.smi_encoder(smi)
        smi_fea = F.normalize(smi_fea, p=2, dim=-1).mean(dim=2)
        splits = torch.split(feature, self.feature_dim_list, dim=-1)
        processed = [mlp(splits[idx]) for idx, mlp in enumerate(self.feature_process)]
        concat_fea = torch.cat(processed, dim=-1)
        gnn_out = self.gnn_encoder(graph).squeeze().reshape(protein_fea.shape)
        fused_x = torch.cat([protein_fea, smi_fea, concat_fea, gnn_out], dim=1)
        self.latent = fused_x
        x = F.relu(self.dropout1(self.fc1(fused_x)))
        x = F.relu(self.dropout2(self.fc2(x)))
        return self.fc(x).squeeze(-1)

    def encode(self):
        return self.latent
