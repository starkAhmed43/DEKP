from pathlib import Path
from typing import Sequence, Tuple
import sys

import torch
import torch_geometric


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CODE_DIR = REPO_ROOT / "DEKP"
if str(ORIGINAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_CODE_DIR))

# Reuse the original DEKP model code directly so bench runs preserve the
# original graph-encoder behavior instead of a bench-specific reimplementation.
from pretrain import (  # noqa: E402
    CNN,
    Context,
    EdgeMLP,
    FeedForward,
    GATLayer,
    GraphEncoder,
    HighwayMLP,
    MLP,
    MetaDecoder,
    ResidualMLP,
    SeqEncoder,
)


def graph_collate_fn(batch: Sequence[Tuple]):
    graph_batch = torch_geometric.data.Batch.from_data_list([item[0] for item in batch])
    protein_tokens = torch.stack([item[1] for item in batch], dim=0)
    smiles_tokens = torch.stack([item[2] for item in batch], dim=0)
    features = torch.stack([item[3] for item in batch], dim=0)
    labels = torch.stack([item[4] for item in batch], dim=0)
    metadata = [item[5] for item in batch]
    return graph_batch, protein_tokens, smiles_tokens, features, labels, metadata
