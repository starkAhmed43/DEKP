import argparse
import ast
import concurrent.futures
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    COMMON_SEQUENCE_COLS,
    DEFAULT_BASE_DIR,
    DEFAULT_SPLIT_GROUPS,
    discover_split_jobs,
    find_first_existing_column,
    normalize_threshold_args,
    read_table,
)


PDB_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9])([0-9][A-Za-z0-9]{3})(?![A-Za-z0-9])")
ALIGNER = PairwiseAligner(mode="global")
ALIGNER.match_score = 1.0
ALIGNER.mismatch_score = 0.0
ALIGNER.open_gap_score = 0.0
ALIGNER.extend_gap_score = 0.0
HELPER_COLUMNS = {
    "original_structure_record",
    "resolved_structure_status",
    "resolved_structure_identity",
    "resolved_structure_chain_id",
}

WORKER_PDB_INDEX = {}
WORKER_PDB_SEQUENCE_CACHE = {}
WORKER_IDENTITY_THRESHOLD = 90.0


def _normalize_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _normalize_sequence_for_alignment(sequence: str) -> str:
    return _normalize_text(sequence).upper().replace("*", "")


def _load_json(path: Path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(path) + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _normalize_group_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    return value


def _extract_pdb_candidates(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_items = [str(item) for item in value]
    else:
        text = _normalize_text(value)
        if not text:
            return []
        if text[0] in "[(" and text[-1] in "])":
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                raw_items = [str(item) for item in parsed]
            else:
                raw_items = re.split(r"[,\s;]+", text)
        else:
            raw_items = re.split(r"[,\s;]+", text)

    candidates: List[str] = []
    seen = set()
    for item in raw_items:
        token = Path(_normalize_text(item)).stem.upper()
        if not token:
            continue
        if re.fullmatch(r"[0-9A-Z]{4}", token):
            if token not in seen:
                seen.add(token)
                candidates.append(token)
            continue
        for match in PDB_ID_PATTERN.findall(token):
            if match not in seen:
                seen.add(match)
                candidates.append(match)
    return candidates


def _build_experimental_index(directory: str) -> Dict[str, str]:
    pdb_dir = Path(directory).expanduser()
    if not pdb_dir.exists():
        raise FileNotFoundError(f"Experimental PDB directory not found: {pdb_dir}")
    index: Dict[str, str] = {}
    for path in pdb_dir.glob("*.pdb"):
        index[path.stem.upper()] = str(path)
    return index


def _extract_sequences_from_pdb(path: str) -> List[dict]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(Path(path).stem, path)
    entries: List[dict] = []
    for model in structure:
        for chain in model:
            residues = []
            for residue in chain:
                if not is_aa(residue, standard=False):
                    continue
                residues.append(seq1(residue.get_resname(), custom_map={"MSE": "M"}, undef_code="X"))
            if residues:
                entries.append(
                    {
                        "chain_id": str(chain.id),
                        "sequence": "".join(residues),
                    }
                )
        break
    return entries


def _prefill_pdb_sequence_cache(
    pending_tasks: List[dict],
    pdb_index: dict,
    pdb_sequence_cache: dict,
    num_threads: int = 8,
) -> None:
    needed = set()
    for task in pending_tasks:
        for pdb_id in task["pdb_candidates"]:
            if pdb_id not in pdb_sequence_cache and pdb_id in pdb_index:
                needed.add(pdb_id)
    if not needed:
        return

    def _parse_one(pdb_id: str):
        return pdb_id, _extract_sequences_from_pdb(pdb_index[pdb_id])

    n_threads = min(num_threads, len(needed))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(_parse_one, pid): pid for pid in needed}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Pre-parsing PDB sequences",
            unit="pdb",
        ):
            pdb_id, entries = future.result()
            pdb_sequence_cache[pdb_id] = entries


def _globalxx_identity_pct(seq1: str, seq2: str) -> float:
    if not seq1 or not seq2:
        return 0.0
    score = ALIGNER.score(seq1, seq2)
    return 100.0 * float(score) / max(len(seq1), len(seq2))


def _init_worker(pdb_index: dict, pdb_sequence_cache: dict, identity_threshold: float) -> None:
    global WORKER_PDB_INDEX, WORKER_PDB_SEQUENCE_CACHE, WORKER_IDENTITY_THRESHOLD
    WORKER_PDB_INDEX = dict(pdb_index)
    WORKER_PDB_SEQUENCE_CACHE = dict(pdb_sequence_cache)
    WORKER_IDENTITY_THRESHOLD = float(identity_threshold)


def _resolve_task(task: dict) -> dict:
    pdb_sequence_updates = {}
    candidates = list(task["pdb_candidates"])
    result = {
        "task_key": task["task_key"],
        "selected_pdb": None,
        "selected_chain_id": "",
        "identity_pct": None,
        "status": "",
        "pdb_sequence_updates": pdb_sequence_updates,
    }

    if not candidates:
        result["status"] = "experimental_no_candidate_ids"
        return result

    if len(candidates) == 1:
        selected = candidates[0]
        if selected in WORKER_PDB_INDEX:
            result["selected_pdb"] = selected
            result["status"] = "single_experimental_candidate"
        else:
            result["status"] = "single_candidate_missing_local_pdb"
        return result

    target_sequence = task["sequence"]
    best_pdb = None
    best_chain_id = ""
    best_identity = -1.0
    seen_sequences = set()

    for pdb_id in candidates:
        pdb_path = WORKER_PDB_INDEX.get(pdb_id)
        if pdb_path is None:
            continue
        chain_entries = WORKER_PDB_SEQUENCE_CACHE.get(pdb_id)
        if chain_entries is None:
            chain_entries = _extract_sequences_from_pdb(pdb_path)
            WORKER_PDB_SEQUENCE_CACHE[pdb_id] = chain_entries
            pdb_sequence_updates[pdb_id] = chain_entries
        for chain_entry in chain_entries:
            pdb_sequence = _normalize_sequence_for_alignment(chain_entry["sequence"])
            if not pdb_sequence or pdb_sequence in seen_sequences:
                continue
            seen_sequences.add(pdb_sequence)
            identity_pct = _globalxx_identity_pct(target_sequence, pdb_sequence)
            if identity_pct > best_identity:
                best_identity = identity_pct
                best_pdb = pdb_id
                best_chain_id = str(chain_entry["chain_id"])

    if best_pdb is None:
        result["status"] = "experimental_candidates_missing_local_pdb"
        return result

    result["identity_pct"] = round(best_identity, 4)
    if best_identity >= WORKER_IDENTITY_THRESHOLD:
        result["selected_pdb"] = best_pdb
        result["selected_chain_id"] = best_chain_id
        result["status"] = "selected_experimental"
    else:
        result["status"] = "experimental_below_identity_threshold"
    return result


def _resolve_columns(frame: pd.DataFrame, args) -> dict:
    return {
        "sequence_col": find_first_existing_column(frame, COMMON_SEQUENCE_COLS, explicit=args.sequence_col, required=True),
        "pdb_type_col": args.pdb_type_col if args.pdb_type_col and args.pdb_type_col in frame.columns else ("pdb_type" if "pdb_type" in frame.columns else None),
        "pdb_source_col": args.pdb_source_col if args.pdb_source_col and args.pdb_source_col in frame.columns else ("pdb_source" if "pdb_source" in frame.columns else None),
        "pdb_record_col": args.pdb_record_col if args.pdb_record_col and args.pdb_record_col in frame.columns else ("pdbs" if "pdbs" in frame.columns else None),
    }


def _write_table(path: Path, frame: pd.DataFrame) -> None:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
        return
    if suffix == ".csv":
        frame.to_csv(path, index=False)
        return
    if suffix == ".tsv":
        frame.to_csv(path, sep="\t", index=False)
        return
    raise ValueError(f"Unsupported table format for write: {path}")


def _make_backup(original_path: Path, backup_root: Path, base_dir: Path) -> Path:
    relative = original_path.relative_to(base_dir)
    backup_path = backup_root / relative
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        shutil.copy2(original_path, backup_path)
    return backup_path


def _result_for_cache(result: dict) -> dict:
    payload = dict(result)
    payload.pop("pdb_sequence_updates", None)
    return payload


def _apply_result(frame: pd.DataFrame, row_indices: List[int], columns: dict, result: dict) -> None:
    pdb_record_col = columns["pdb_record_col"]
    for idx in row_indices:
        if result["selected_pdb"]:
            frame.at[idx, pdb_record_col] = result["selected_pdb"]
        frame.at[idx, "resolved_structure_status"] = result["status"]
        frame.at[idx, "resolved_structure_identity"] = result["identity_pct"]
        frame.at[idx, "resolved_structure_chain_id"] = result["selected_chain_id"]


def _build_grouped_tasks(frame: pd.DataFrame, columns: dict) -> tuple[dict, int]:
    pdb_record_col = columns["pdb_record_col"]
    pdb_type_col = columns["pdb_type_col"]
    sequence_col = columns["sequence_col"]

    exp_mask = frame[pdb_type_col].fillna("").astype(str).str.strip().str.lower() == "experimental"
    experimental_rows = int(exp_mask.sum())
    if experimental_rows == 0:
        return {}, 0

    group_columns = [
        column
        for column in frame.columns
        if column not in ({pdb_record_col} | HELPER_COLUMNS)
    ]

    grouped_tasks = {}
    for row_index, row in frame.loc[exp_mask].iterrows():
        key_payload = {column: _normalize_group_value(row[column]) for column in group_columns}
        task_key = json.dumps(key_payload, sort_keys=True, default=str)
        task = grouped_tasks.get(task_key)
        if task is None:
            task = {
                "task_key": task_key,
                "sequence": _normalize_sequence_for_alignment(row[sequence_col]),
                "pdb_candidates": set(),
                "row_indices": [],
            }
            grouped_tasks[task_key] = task
        task["row_indices"].append(int(row_index))
        task["pdb_candidates"].update(_extract_pdb_candidates(row[pdb_record_col]))

    for task in grouped_tasks.values():
        task["pdb_candidates"] = tuple(sorted(task["pdb_candidates"]))
    return grouped_tasks, experimental_rows


def _rewrite_split_file(
    path: Path,
    base_dir: Path,
    args,
    experimental_index: dict,
    pdb_sequence_cache: dict,
    selection_cache: dict,
    summary: dict,
) -> None:
    frame = read_table(path)
    columns = _resolve_columns(frame, args)
    if columns["pdb_type_col"] is None or columns["pdb_record_col"] is None:
        print(f"Skipping {path}: required pdb columns were not found.", flush=True)
        return

    if "original_structure_record" not in frame.columns:
        frame["original_structure_record"] = frame[columns["pdb_record_col"]]
    if "resolved_structure_status" not in frame.columns:
        frame["resolved_structure_status"] = ""
    if "resolved_structure_identity" not in frame.columns:
        frame["resolved_structure_identity"] = pd.NA
    if "resolved_structure_chain_id" not in frame.columns:
        frame["resolved_structure_chain_id"] = ""

    grouped_tasks, experimental_rows = _build_grouped_tasks(frame, columns)
    tasks = list(grouped_tasks.values())
    cached_task_count = 0
    pending_tasks = []
    for task in tasks:
        cached = selection_cache.get(task["task_key"])
        if cached is not None:
            task["result"] = cached
            cached_task_count += 1
        else:
            pending_tasks.append(
                {
                    "task_key": task["task_key"],
                    "sequence": task["sequence"],
                    "pdb_candidates": task["pdb_candidates"],
                }
            )

    if pending_tasks:
        # Pre-parse all PDB sequences referenced by pending tasks into the shared
        # cache using threads (I/O-bound). Workers then inherit the full cache via
        # fork and do zero disk I/O during alignment.
        _prefill_pdb_sequence_cache(pending_tasks, experimental_index, pdb_sequence_cache, num_threads=min(args.num_workers, 16))
        # Populate globals in parent; forked workers inherit them instantly.
        _init_worker(experimental_index, pdb_sequence_cache, args.identity_threshold)
        iterator = tqdm(total=len(pending_tasks), desc=f"Resolving {path.relative_to(base_dir)}", unit="task", leave=True)
        if args.num_workers > 1:
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=args.num_workers) as pool:
                for result in pool.imap_unordered(_resolve_task, pending_tasks, chunksize=1):
                    selection_cache[result["task_key"]] = _result_for_cache(result)
                    grouped_tasks[result["task_key"]]["result"] = result
                    iterator.update(1)
        else:
            for task in pending_tasks:
                result = _resolve_task(task)
                selection_cache[result["task_key"]] = _result_for_cache(result)
                grouped_tasks[result["task_key"]]["result"] = result
                iterator.update(1)
        iterator.close()

    changed_rows = 0
    selected_rows = 0
    unresolved_rows = 0
    skipped_single_rows = 0
    for task in tasks:
        result = task["result"]
        _apply_result(frame, task["row_indices"], columns, result)
        row_count = len(task["row_indices"])
        if result["status"] in {"selected_experimental", "single_experimental_candidate"}:
            changed_rows += row_count
            if result["status"] == "selected_experimental":
                selected_rows += row_count
            else:
                skipped_single_rows += row_count
        else:
            unresolved_rows += row_count

    summary["files"].append(
        {
            "path": str(path),
            "rows": int(len(frame)),
            "experimental_rows": int(experimental_rows),
            "unique_resolution_tasks": int(len(tasks)),
            "cached_resolution_tasks": int(cached_task_count),
            "changed_rows": int(changed_rows),
            "selected_experimental_rows": int(selected_rows),
            "single_candidate_rows": int(skipped_single_rows),
            "unresolved_rows": int(unresolved_rows),
        }
    )

    if args.dry_run:
        return

    backup_path = _make_backup(path, args.backup_dir, base_dir)
    _write_table(path, frame)
    print(f"Rewrote {path} (backup: {backup_path})", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve experimental PDB choices across split files once, using only local experimental PDBs.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--sequence_col", type=str, default=None)
    parser.add_argument("--pdb_type_col", type=str, default=None)
    parser.add_argument("--pdb_source_col", type=str, default=None)
    parser.add_argument("--pdb_record_col", type=str, default=None)
    parser.add_argument("--experimental_pdb_dir", type=str, default="~/github/EMULaToR/data/intermediate/processed_exp_pdb")
    parser.add_argument("--identity_threshold", type=float, default=90.0)
    parser.add_argument("--pdb_sequence_cache_json", type=str, default=None)
    parser.add_argument("--selection_cache_json", type=str, default=None)
    parser.add_argument("--backup_dir", type=str, default=None)
    parser.add_argument("--summary_json", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 1) - 5))
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    base_dir = Path(args.base_dir).expanduser()

    if args.backup_dir is None:
        args.backup_dir = base_dir / "_pdb_resolution_backups"
    else:
        args.backup_dir = Path(args.backup_dir).expanduser()

    if args.pdb_sequence_cache_json is None:
        args.pdb_sequence_cache_json = args.backup_dir / "pdb_sequence_cache.json"
    else:
        args.pdb_sequence_cache_json = Path(args.pdb_sequence_cache_json).expanduser()

    if args.selection_cache_json is None:
        args.selection_cache_json = args.backup_dir / "selection_cache.json"
    else:
        args.selection_cache_json = Path(args.selection_cache_json).expanduser()

    if args.summary_json is None:
        args.summary_json = args.backup_dir / "rewrite_summary.json"
    else:
        args.summary_json = Path(args.summary_json).expanduser()

    jobs = discover_split_jobs(base_dir, split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {base_dir}")

    split_files: List[Path] = []
    seen = set()
    for job in jobs:
        for key in ["train_path", "val_path", "test_path"]:
            path = Path(job[key])
            if str(path) not in seen:
                seen.add(str(path))
                split_files.append(path)

    experimental_index = _build_experimental_index(args.experimental_pdb_dir)
    pdb_sequence_cache = _load_json(Path(args.pdb_sequence_cache_json), {})
    selection_cache = _load_json(Path(args.selection_cache_json), {})

    summary = {
        "base_dir": str(base_dir),
        "split_groups": list(args.split_groups),
        "thresholds": args.thresholds,
        "identity_threshold": float(args.identity_threshold),
        "dry_run": bool(args.dry_run),
        "files": [],
    }

    for path in split_files:
        _rewrite_split_file(
            path=path,
            base_dir=base_dir,
            args=args,
            experimental_index=experimental_index,
            pdb_sequence_cache=pdb_sequence_cache,
            selection_cache=selection_cache,
            summary=summary,
        )

    _save_json(Path(args.pdb_sequence_cache_json), pdb_sequence_cache)
    _save_json(Path(args.selection_cache_json), selection_cache)

    aggregate = {
        "files_processed": len(summary["files"]),
        "experimental_rows": sum(item["experimental_rows"] for item in summary["files"]),
        "unique_resolution_tasks": sum(item["unique_resolution_tasks"] for item in summary["files"]),
        "cached_resolution_tasks": sum(item["cached_resolution_tasks"] for item in summary["files"]),
        "changed_rows": sum(item["changed_rows"] for item in summary["files"]),
        "selected_experimental_rows": sum(item["selected_experimental_rows"] for item in summary["files"]),
        "single_candidate_rows": sum(item["single_candidate_rows"] for item in summary["files"]),
        "unresolved_rows": sum(item["unresolved_rows"] for item in summary["files"]),
    }
    summary["aggregate"] = aggregate
    _save_json(Path(args.summary_json), summary)
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
