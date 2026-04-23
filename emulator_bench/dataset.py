from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import math
import pandas as pd
import torch

from emulator_bench.common import (
    ligand_cache_path,
    normalize_sequence,
    protein_cache_path,
    protein_sequence_cache_max_len,
    structure_cache_path,
)


def _load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load cached payload {path}. Rebuild this cache entry or rerun cache_embeddings.py --overwrite."
        ) from exc


class _LRUStore:
    def __init__(self, max_items: int = 256):
        self.max_items = max(1, int(max_items))
        self._cache: "OrderedDict[str, object]" = OrderedDict()

    def _get_or_load(self, key: str, path: Path):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        payload = _load_pt(path)
        self._cache[key] = payload
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return payload


class ProteinEmbeddingStore(_LRUStore):
    def __init__(self, cache_dir: Path, sequences: Optional[Iterable[str]] = None, preload: bool = False, max_items: int = 256, max_len: int = 2500):
        super().__init__(max_items=max_items)
        self.cache_dir = Path(cache_dir)
        self.max_len = max_len
        self.cache_key_max_len = protein_sequence_cache_max_len(max_len)
        if preload and sequences is not None:
            for sequence in sorted({normalize_sequence(sequence, max_len=self.cache_key_max_len) for sequence in sequences}):
                self.get(sequence)

    def get(self, sequence: str):
        normalized = normalize_sequence(sequence, max_len=self.cache_key_max_len)
        path = protein_cache_path(self.cache_dir, normalized, max_len=self.cache_key_max_len)
        if not path.exists():
            raise FileNotFoundError(f"Missing cached protein payload: {path}")
        return self._get_or_load(normalized, path)


class LigandEmbeddingStore(_LRUStore):
    def __init__(self, cache_dir: Path, smiles_values: Optional[Iterable[str]] = None, preload: bool = False, max_items: int = 256):
        super().__init__(max_items=max_items)
        self.cache_dir = Path(cache_dir)
        if preload and smiles_values is not None:
            for smiles in sorted({str(smiles) for smiles in smiles_values}):
                self.get(smiles)

    def get(self, smiles: str):
        smiles = str(smiles).strip()
        path = ligand_cache_path(self.cache_dir, smiles)
        if not path.exists():
            raise FileNotFoundError(f"Missing cached ligand payload: {path}")
        return self._get_or_load(smiles, path)


class StructureEmbeddingStore(_LRUStore):
    def __init__(self, cache_dir: Path, items: Optional[Iterable[tuple[str, str]]] = None, preload: bool = False, max_items: int = 256):
        super().__init__(max_items=max_items)
        self.cache_dir = Path(cache_dir)
        if preload and items is not None:
            seen = set()
            for structure_id, sequence in items:
                key = f"{structure_id}::{normalize_sequence(sequence)}"
                if key in seen:
                    continue
                seen.add(key)
                self.get(structure_id, sequence)

    def get(self, structure_id: str, sequence: str):
        path = structure_cache_path(self.cache_dir, structure_id=structure_id, fallback_sequence=sequence)
        if not path.exists():
            raise FileNotFoundError(f"Missing cached structure payload: {path}")
        cache_key = f"{structure_id}::{normalize_sequence(sequence)}"
        return self._get_or_load(cache_key, path)


def pad_tokens(token_ids: torch.Tensor, max_length: int, pad_id: int) -> torch.Tensor:
    token_ids = token_ids[:max_length]
    if token_ids.numel() < max_length:
        padding = torch.full((max_length - token_ids.numel(),), int(pad_id), dtype=torch.long)
        token_ids = torch.cat([token_ids.long(), padding], dim=0)
    return token_ids.long()


class CachedDEKPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        protein_store: ProteinEmbeddingStore,
        ligand_store: LigandEmbeddingStore,
        structure_store: StructureEmbeddingStore,
        feature_names: List[str],
        sequence_col: str,
        smiles_col: str,
        protein_id_col: str,
        target_col: Optional[str],
        protein_max_len: int,
        smiles_max_len: int,
        protein_pad_id: int = 0,
        smiles_pad_id: int = 0,
        structure_id_col: Optional[str] = None,
    ):
        super().__init__()
        self.frame = frame.reset_index(drop=True)
        self.protein_store = protein_store
        self.ligand_store = ligand_store
        self.structure_store = structure_store
        self.feature_names = list(feature_names)
        self.sequence_col = sequence_col
        self.smiles_col = smiles_col
        self.protein_id_col = protein_id_col
        self.structure_id_col = structure_id_col or protein_id_col
        self.target_col = target_col
        self.protein_max_len = int(protein_max_len)
        self.smiles_max_len = int(smiles_max_len)
        self.protein_pad_id = int(protein_pad_id)
        self.smiles_pad_id = int(smiles_pad_id)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        sequence = str(row[self.sequence_col])
        smiles = str(row[self.smiles_col])
        protein_id = str(row[self.protein_id_col])
        raw_struct = row[self.structure_id_col] if self.structure_id_col in row.index else None
        if raw_struct is None or (isinstance(raw_struct, float) and math.isnan(raw_struct)) or str(raw_struct).strip().lower() in ("", "nan", "none"):
            structure_id = protein_id
        else:
            structure_id = str(raw_struct).strip()

        protein_payload = self.protein_store.get(sequence)
        ligand_payload = self.ligand_store.get(smiles)
        structure_payload = self.structure_store.get(structure_id, sequence)

        feature_tensors = []
        for feature_name in self.feature_names:
            if feature_name in protein_payload:
                feature_tensors.append(protein_payload[feature_name].float())
            elif feature_name in ligand_payload:
                feature_tensors.append(ligand_payload[feature_name].float())
            elif feature_name in structure_payload:
                feature_tensors.append(structure_payload[feature_name].float())
            else:
                raise KeyError(
                    f"Feature `{feature_name}` was requested but is missing from cached protein/ligand/structure payloads."
                )
        feature_tensor = torch.cat(feature_tensors, dim=-1).float()

        label_value = float(row[self.target_col]) if self.target_col and self.target_col in row.index else float("nan")
        label_tensor = torch.tensor(label_value, dtype=torch.float32)
        protein_tokens = pad_tokens(protein_payload["token_ids"].long(), self.protein_max_len, self.protein_pad_id)
        smiles_tokens = pad_tokens(ligand_payload["token_ids"].long(), self.smiles_max_len, self.smiles_pad_id)
        graph = structure_payload["graph"]

        metadata = {
            "row_index": int(idx),
            "protein_id": protein_id,
            "structure_id": structure_id,
            "sequence": sequence,
            "smiles": smiles,
            "target": label_value,
        }
        for optional_key in ["ECNumber", "ec_number", "Type", "type", "uniprot_id", "UniprotID"]:
            if optional_key in row.index:
                metadata[optional_key] = row[optional_key]

        return graph, protein_tokens, smiles_tokens, feature_tensor, label_tensor, metadata
