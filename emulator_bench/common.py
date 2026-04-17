import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_BASE_DIR = Path("~/github/EMULaToR/data/processed/baselines/DEKP").expanduser()
DEFAULT_CACHE_DIR = DEFAULT_BASE_DIR / "embeddings"
DEFAULT_SPLIT_GROUPS = [
    "random_splits",
    "enzyme_sequence_splits",
    "substrate_splits",
    "group_shuffle_splits",
]
DEFAULT_FEATURES = ["trfm", "t5"]

COMMON_SEQUENCE_COLS = ["sequence", "Sequence"]
COMMON_SMILES_COLS = ["smiles", "Smiles"]
COMMON_TARGET_COLS = ["log10_value", "Label", "value"]
COMMON_PROTEIN_ID_COLS = ["uniprot_id", "UniprotID", "Protein", "protein_id"]
COMMON_STRUCTURE_ID_COLS = ["structure_id", "pdb_id", "uniprot_id", "UniprotID"]
COMMON_CID_COLS = ["CID", "cid"]


def _stable_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def normalize_sequence(sequence: str, max_len: int = 2500) -> str:
    seq = str(sequence).strip().upper()[:max_len]
    return seq.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X").replace("*", "")


def normalize_smiles(smiles: str) -> str:
    return str(smiles).strip()


def protein_cache_key(sequence: str, max_len: int = 2500) -> str:
    return _stable_hash(normalize_sequence(sequence, max_len=max_len))


def ligand_cache_key(smiles: str) -> str:
    return _stable_hash(normalize_smiles(smiles))


def structure_cache_key(structure_id: str, fallback_sequence: str) -> str:
    structure_id = str(structure_id).strip()
    if structure_id:
        return _stable_hash(structure_id)
    return protein_cache_key(fallback_sequence)


def protein_cache_path(cache_dir: Path, sequence: str, max_len: int = 2500) -> Path:
    key = protein_cache_key(sequence, max_len=max_len)
    return Path(cache_dir) / "proteins" / key[:2] / f"{key}.pt"


def ligand_cache_path(cache_dir: Path, smiles: str) -> Path:
    key = ligand_cache_key(smiles)
    return Path(cache_dir) / "ligands" / key[:2] / f"{key}.pt"


def structure_cache_path(cache_dir: Path, structure_id: str, fallback_sequence: str) -> Path:
    key = structure_cache_key(structure_id, fallback_sequence)
    return Path(cache_dir) / "structures" / key[:2] / f"{key}.pt"


def tokenizers_dir(cache_dir: Path) -> Path:
    return Path(cache_dir) / "tokenizers"


def protein_vocab_path(cache_dir: Path) -> Path:
    return tokenizers_dir(cache_dir) / "protein_vocab.json"


def smiles_vocab_path(cache_dir: Path) -> Path:
    return tokenizers_dir(cache_dir) / "smiles_vocab.json"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict) -> None:
    ensure_parent(path)
    tmp_path = Path(str(path) + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported table format: {path}")


def require_columns(df: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")


def find_first_existing_column(df: pd.DataFrame, candidates: Iterable[str], explicit: Optional[str] = None, required: bool = True) -> Optional[str]:
    if explicit:
        if explicit in df.columns:
            return explicit
        if required:
            raise ValueError(f"Requested column `{explicit}` not found. Available columns: {list(df.columns)}")
        return None
    for name in candidates:
        if name in df.columns:
            return name
    if required:
        raise ValueError(f"Could not resolve a required column from candidates: {list(candidates)}")
    return None


def _threshold_value(name: str) -> float:
    try:
        return float(name.split("threshold_")[-1])
    except Exception:
        return math.inf


def _difficulty_labels_for_thresholds(names: List[str]) -> Dict[str, str]:
    ordered = sorted(names, key=_threshold_value)
    if len(ordered) == 1:
        return {ordered[0]: "single"}
    if len(ordered) == 2:
        return {ordered[0]: "hard", ordered[1]: "easy"}
    if len(ordered) == 3:
        return {ordered[0]: "hard", ordered[1]: "medium", ordered[2]: "easy"}
    return {name: f"rank_{idx + 1}" for idx, name in enumerate(ordered)}


def normalize_threshold_args(
    thresholds: Optional[Iterable[str]] = None,
    threshold: Optional[str] = None,
) -> Optional[List[str]]:
    values: List[str] = []
    if thresholds is not None:
        values.extend([str(value) for value in thresholds if str(value).strip()])
    if threshold is not None and str(threshold).strip():
        values.append(str(threshold))
    if not values:
        return None
    deduped: List[str] = []
    seen = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _find_split_file(directory: Path, stem: str) -> Optional[Path]:
    for suffix in (".parquet", ".csv", ".tsv"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def discover_split_jobs(
    base_dir: Path,
    split_groups: Optional[Iterable[str]] = None,
    thresholds: Optional[Iterable[str]] = None,
) -> List[Dict[str, str]]:
    split_groups = list(split_groups or DEFAULT_SPLIT_GROUPS)
    threshold_filter = list(thresholds) if thresholds is not None else None
    jobs: List[Dict[str, str]] = []

    for split_group in split_groups:
        group_dir = Path(base_dir) / split_group
        if not group_dir.exists():
            continue

        if split_group in {"random_splits", "group_shuffle_splits"}:
            train_path = _find_split_file(group_dir, "train")
            val_path = _find_split_file(group_dir, "val")
            test_path = _find_split_file(group_dir, "test")
            if train_path and val_path and test_path:
                split_name = "random" if split_group == "random_splits" else "group_shuffle"
                jobs.append(
                    {
                        "split_group": split_group,
                        "split_name": split_name,
                        "difficulty": split_name,
                        "root_dir": str(group_dir),
                        "train_path": str(train_path),
                        "val_path": str(val_path),
                        "test_path": str(test_path),
                    }
                )
            continue

        candidate_dirs = []
        for child in sorted(group_dir.iterdir()):
            if not child.is_dir():
                continue
            if threshold_filter is not None and child.name not in threshold_filter:
                continue
            if child.name.startswith("threshold_") or child.name in {"easy", "medium", "hard"}:
                candidate_dirs.append(child)

        threshold_names = [path.name for path in candidate_dirs if path.name.startswith("threshold_")]
        threshold_difficulties = _difficulty_labels_for_thresholds(threshold_names)

        for child in candidate_dirs:
            train_path = _find_split_file(child, "train")
            val_path = _find_split_file(child, "val")
            test_path = _find_split_file(child, "test")
            if not (train_path and val_path and test_path):
                continue

            difficulty = threshold_difficulties.get(child.name, child.name)
            jobs.append(
                {
                    "split_group": split_group,
                    "split_name": child.name,
                    "difficulty": difficulty,
                    "root_dir": str(child),
                    "train_path": str(train_path),
                    "val_path": str(val_path),
                    "test_path": str(test_path),
                }
            )

    return jobs


def resolve_single_split_job(base_dir: Path, split_group: str, threshold: Optional[str] = None) -> Dict[str, str]:
    threshold_filter = None if split_group in {"random_splits", "group_shuffle_splits"} else normalize_threshold_args(threshold=threshold)
    jobs = discover_split_jobs(base_dir, split_groups=[split_group], thresholds=threshold_filter)
    if not jobs:
        detail = f"{split_group}/{threshold}" if threshold else split_group
        raise FileNotFoundError(f"No split job discovered for {detail} in {base_dir}")
    if split_group in {"random_splits", "group_shuffle_splits"}:
        return jobs[0]
    if threshold is None and len(jobs) > 1:
        available = ", ".join(job["split_name"] for job in jobs)
        raise ValueError(f"Multiple jobs found for {split_group}. Specify --threshold. Available: {available}")
    if threshold is None:
        return jobs[0]
    matching = [job for job in jobs if job["split_name"] == threshold]
    if not matching:
        available = ", ".join(job["split_name"] for job in jobs)
        raise FileNotFoundError(f"Threshold `{threshold}` not found for {split_group}. Available: {available}")
    return matching[0]


def split_sizes(train_path: Path, val_path: Path, test_path: Path) -> Dict[str, float]:
    train_size = len(read_table(train_path))
    val_size = len(read_table(val_path))
    test_size = len(read_table(test_path))
    total = train_size + val_size + test_size
    if total == 0:
        return {
            "train_size": 0,
            "val_size": 0,
            "test_size": 0,
            "train_ratio": 0.0,
            "val_ratio": 0.0,
            "test_ratio": 0.0,
        }
    return {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "train_ratio": train_size / total,
        "val_ratio": val_size / total,
        "test_ratio": test_size / total,
    }


def summarize_seed_runs(rows: List[Dict], group_cols: Iterable[str], metric_cols: Iterable[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    runs_df = pd.DataFrame(rows)
    out_rows = []
    for keys, group in runs_df.groupby(list(group_cols), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group.columns else len(group)
        for col in runs_df.columns:
            if col in row or col in metric_cols or col == "seed" or col.endswith("_dir"):
                continue
            values = group[col].dropna()
            if len(values) > 0:
                row[col] = values.iloc[0]
        for metric in metric_cols:
            if metric not in group.columns:
                continue
            values = group[metric].dropna()
            if len(values) == 0:
                continue
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_var"] = float(values.var(ddof=1)) if len(values) > 1 else 0.0
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def write_csv(path: Path, rows: List[Dict]) -> None:
    ensure_parent(path)
    pd.DataFrame(rows).to_csv(path, index=False)


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)

    unique_vals, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for group_idx, count in enumerate(counts):
        if count <= 1:
            continue
        positions = np.where(inverse == group_idx)[0]
        mean_rank = ranks[positions].mean()
        ranks[positions] = mean_rank
    return ranks


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) == 0:
        return {
            "rmse": float("nan"),
            "mse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
        }

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    total_var = float(np.sum((y_true - np.mean(y_true)) ** 2))
    residual = float(np.sum((y_true - y_pred) ** 2))
    r2 = float("nan") if total_var == 0 else 1.0 - residual / total_var
    pearson = _safe_corrcoef(y_true, y_pred)
    spearman = _safe_corrcoef(_rankdata(y_true), _rankdata(y_pred))
    return {
        "rmse": float(math.sqrt(mse)),
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
    }


def metric_sort_ascending(metric: str) -> bool:
    return metric in {"rmse", "mse", "mae"}


def resolve_metric_value(metrics: Dict[str, float], metric_name: str) -> float:
    aliases = {"r2_score": "r2"}
    key = aliases.get(metric_name, metric_name)
    if key not in metrics:
        raise KeyError(f"Metric `{metric_name}` not found in metrics: {sorted(metrics)}")
    return float(metrics[key])


def append_csv_row(path: Path, row: Dict[str, object]) -> None:
    ensure_parent(path)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
