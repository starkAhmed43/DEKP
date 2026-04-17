import argparse
import gc
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    COMMON_CID_COLS,
    COMMON_PROTEIN_ID_COLS,
    COMMON_SEQUENCE_COLS,
    COMMON_SMILES_COLS,
    COMMON_STRUCTURE_ID_COLS,
    DEFAULT_BASE_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_FEATURES,
    DEFAULT_SPLIT_GROUPS,
    discover_split_jobs,
    find_first_existing_column,
    normalize_sequence,
    normalize_threshold_args,
    protein_cache_path,
    ligand_cache_path,
    save_json,
    structure_cache_path,
    tokenizers_dir,
    protein_vocab_path,
    read_table,
    smiles_vocab_path,
)
from emulator_bench.feature_pipeline import (
    build_graph_from_pdb,
    build_graph_from_array,
    build_prot_t5_batches,
    compute_dssp_feature,
    embed_prot_t5_batch,
    embed_smiles_trfm_batch,
    load_legacy_feature_pickles,
    load_prot_t5,
    load_smiles_transformer,
    parse_pdb_to_array,
    resolve_legacy_feature,
)
from emulator_bench.bench_tokenizers import ProteinSequenceTokenizer, RegexSmilesTokenizer


DEFAULT_DATASET_DF_PATH = Path("~/github/EMULaToR/data/processed/baselines/DEKP/km_kinetic_params_3d.parquet").expanduser()


def _log(message: str, enabled: bool = True) -> None:
    if enabled:
        print(message, flush=True)


def _atomic_torch_save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(path) + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _coerce_feature_tensor(value, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    else:
        tensor = torch.as_tensor(np.asarray(value))
    if tensor.ndim > 1:
        tensor = tensor.float().mean(dim=0)
    return tensor.reshape(-1).to(dtype=dtype)


def _canonical_pdb_identity(pdb_type, pdb_source, pdb_record, structure_id, protein_id):
    pdb_type_value = str(pdb_type or "").strip().lower()
    pdb_source_value = str(pdb_source or "").strip().lower()
    pdb_record_value = str(pdb_record or "").strip()
    structure_id_value = str(structure_id or "").strip()
    protein_id_value = str(protein_id or "").strip()
    pdb_key = pdb_record_value or structure_id_value or protein_id_value
    if not pdb_key:
        return None
    return f"{pdb_type_value}::{pdb_source_value}::{pdb_key}"


def _normalize_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _build_pdb_indexes(args) -> dict:
    indexes = {
        "experimental": {},
        "alphafold": {},
        "esm3": {},
    }

    if args.experimental_pdb_dir:
        pdb_dir = Path(args.experimental_pdb_dir).expanduser()
        if pdb_dir.exists():
            for path in pdb_dir.glob(f"*{args.pdb_suffix}"):
                indexes["experimental"][path.name] = path
                indexes["experimental"][path.stem] = path

    if args.alphafold_pdb_dir:
        pdb_dir = Path(args.alphafold_pdb_dir).expanduser()
        if pdb_dir.exists():
            for path in pdb_dir.glob("AF-*-F1-model_v*.pdb"):
                stem = path.stem
                prefix = "AF-"
                middle = "-F1-model_v"
                if stem.startswith(prefix) and middle in stem:
                    uniprot_id, _, version_text = stem[len(prefix):].partition(middle)
                    try:
                        version = int(version_text)
                    except ValueError:
                        version = -1
                    current = indexes["alphafold"].get(uniprot_id)
                    if current is None or version > current[0]:
                        indexes["alphafold"][uniprot_id] = (version, path)

    if args.esm3_pdb_dir:
        pdb_dir = Path(args.esm3_pdb_dir).expanduser()
        if pdb_dir.exists():
            for path in pdb_dir.glob("ESM3-open-small-*.pdb"):
                key = path.name[len("ESM3-open-small-") : -len(".pdb")]
                indexes["esm3"][key] = path

    return indexes


def _build_existing_structure_cache_index(cache_dir: Path) -> set[str]:
    structures_dir = Path(cache_dir) / "structures"
    if not structures_dir.exists():
        return set()
    return {str(path) for path in structures_dir.rglob("*.pt")}


def _iter_unique_rows_from_frame(frame: pd.DataFrame, column_names: dict):
    sequence_col = column_names["sequence_col"]
    smiles_col = column_names["smiles_col"]
    protein_id_col = column_names.get("protein_id_col")
    structure_id_col = column_names.get("structure_id_col")
    cid_col = column_names.get("cid_col")
    pdb_type_col = column_names.get("pdb_type_col")
    pdb_source_col = column_names.get("pdb_source_col")
    pdb_record_col = column_names.get("pdb_record_col")

    working = pd.DataFrame(index=frame.index)
    working["sequence"] = _normalize_series(frame[sequence_col]).map(normalize_sequence)
    working["smiles"] = _normalize_series(frame[smiles_col])

    if protein_id_col and protein_id_col in frame.columns:
        working["protein_id"] = _normalize_series(frame[protein_id_col])
    else:
        working["protein_id"] = working["sequence"]

    if structure_id_col and structure_id_col in frame.columns:
        working["structure_id"] = _normalize_series(frame[structure_id_col])
    elif pdb_record_col and pdb_record_col in frame.columns:
        working["structure_id"] = _normalize_series(frame[pdb_record_col])
    else:
        working["structure_id"] = working["protein_id"]

    if cid_col and cid_col in frame.columns:
        working["cid"] = frame[cid_col]
    else:
        working["cid"] = None

    if pdb_type_col and pdb_type_col in frame.columns:
        working["pdb_type"] = _normalize_series(frame[pdb_type_col]).str.lower()
    else:
        working["pdb_type"] = ""

    if pdb_source_col and pdb_source_col in frame.columns:
        working["pdb_source"] = _normalize_series(frame[pdb_source_col]).str.lower()
    else:
        working["pdb_source"] = ""

    if pdb_record_col and pdb_record_col in frame.columns:
        working["pdb_record"] = _normalize_series(frame[pdb_record_col])
    else:
        working["pdb_record"] = working["structure_id"]

    working["pdb_key"] = working["pdb_record"].where(working["pdb_record"] != "", working["structure_id"])
    working = working[working["sequence"] != ""].copy()
    working = working[working["smiles"] != ""].copy()
    working = working[working["pdb_key"] != ""].copy()

    unique_sequences = (
        working.drop_duplicates(subset=["sequence"], keep="first")[["sequence", "protein_id", "structure_id"]]
        .to_dict("records")
    )
    unique_smiles = working.drop_duplicates(subset=["smiles"], keep="first")[["smiles", "cid"]].to_dict("records")
    unique_structures = (
        working.drop_duplicates(subset=["pdb_key"], keep="first")[
            ["pdb_key", "sequence", "protein_id", "pdb_type", "pdb_source", "pdb_record", "structure_id"]
        ]
        .to_dict("records")
    )

    unique_sequences_map = {
        entry["sequence"]: {
            "sequence": entry["sequence"],
            "protein_id": entry["protein_id"] or entry["sequence"],
            "structure_id": entry["structure_id"] or entry["sequence"],
        }
        for entry in unique_sequences
    }
    unique_smiles_map = {
        entry["smiles"]: {
            "smiles": entry["smiles"],
            "cid": entry["cid"],
        }
        for entry in unique_smiles
    }
    unique_structures_map = {
        entry["pdb_key"]: {
            "structure_id": entry["pdb_key"],
            "protein_id": entry["protein_id"] or entry["sequence"],
            "sequence": entry["sequence"],
            "pdb_type": entry["pdb_type"],
            "pdb_source": entry["pdb_source"],
            "pdb_record": entry["pdb_record"] or entry["structure_id"] or entry["pdb_key"],
            "original_structure_id": entry["structure_id"] or entry["pdb_record"] or entry["pdb_key"],
        }
        for entry in unique_structures
    }

    protein_len_max = working["sequence"].str.len().max()
    smiles_len_max = working["smiles"].str.len().max()
    max_protein_len = (int(protein_len_max) if pd.notna(protein_len_max) else 0) + 2
    max_smiles_len = (int(smiles_len_max) if pd.notna(smiles_len_max) else 0) + 2
    return unique_sequences_map, unique_smiles_map, unique_structures_map, max_protein_len, max_smiles_len


def _scan_split_file(payload: dict) -> dict:
    frame = read_table(Path(payload["path"]))
    sequence_col = payload["column_names"]["sequence_col"]
    smiles_col = payload["column_names"]["smiles_col"]
    protein_id_col = payload["column_names"]["protein_id_col"]
    structure_id_col = payload["column_names"]["structure_id_col"]
    cid_col = payload["column_names"]["cid_col"]
    pdb_type_col = payload["column_names"]["pdb_type_col"]
    pdb_source_col = payload["column_names"]["pdb_source_col"]
    pdb_record_col = payload["column_names"]["pdb_record_col"]

    unique_sequences = {}
    unique_smiles = {}
    unique_structures = {}
    max_protein_len = 0
    max_smiles_len = 0

    for _, row in frame.iterrows():
        sequence = normalize_sequence(row[sequence_col])
        smiles = str(row[smiles_col]).strip()
        protein_id = str(row[protein_id_col]) if protein_id_col else sequence
        structure_id = str(row[structure_id_col]) if structure_id_col else protein_id
        cid_value = None
        if cid_col and cid_col in row.index:
            cid_value = row[cid_col]
        pdb_type = None
        if pdb_type_col and pdb_type_col in row.index:
            pdb_type = row[pdb_type_col]
        pdb_source = None
        if pdb_source_col and pdb_source_col in row.index:
            pdb_source = row[pdb_source_col]
        pdb_record = None
        if pdb_record_col and pdb_record_col in row.index:
            pdb_record = row[pdb_record_col]
        pdb_identity = _canonical_pdb_identity(pdb_type, pdb_source, pdb_record, structure_id, protein_id)

        if sequence not in unique_sequences:
            unique_sequences[sequence] = {
                "sequence": sequence,
                "protein_id": protein_id,
                "structure_id": structure_id,
            }
        if smiles not in unique_smiles:
            unique_smiles[smiles] = {
                "smiles": smiles,
                "cid": cid_value,
            }
        if pdb_identity and pdb_identity not in unique_structures:
            unique_structures[pdb_identity] = {
                "structure_id": pdb_identity,
                "protein_id": protein_id,
                "sequence": sequence,
                "pdb_type": pdb_type,
                "pdb_source": pdb_source,
                "pdb_record": pdb_record,
                "original_structure_id": structure_id,
            }

        max_protein_len = max(max_protein_len, len(sequence) + 2)
        max_smiles_len = max(max_smiles_len, len(smiles) + 2)

    return {
        "path": payload["path"],
        "split_group": payload["split_group"],
        "split_name": payload["split_name"],
        "rows": len(frame),
        "unique_sequences": unique_sequences,
        "unique_smiles": unique_smiles,
        "unique_structures": unique_structures,
        "max_protein_len": max_protein_len,
        "max_smiles_len": max_smiles_len,
    }


def _resolve_pdb_path(entry: dict, args) -> Path | None:
    pdb_indexes = getattr(args, "_pdb_indexes", None)
    pdb_record_id = str(entry.get("pdb_record") or entry.get("structure_id") or entry.get("protein_id") or "").strip()
    pdb_type = str(entry.get("pdb_type") or "").strip().lower()
    pdb_source = str(entry.get("pdb_source") or "").strip()
    pdb_source_lower = pdb_source.lower()
    pdb_record = str(entry.get("pdb_record") or "").strip()
    if "|" in pdb_record:
        if pdb_indexes is None:
            return None
        return pdb_indexes["esm3"].get(pdb_record)

    if pdb_type == "experimental":
        if pdb_indexes is None:
            return None
        candidates = []
        if pdb_record:
            candidates.append(pdb_record)
            if "." not in Path(pdb_record).name:
                candidates.append(f"{pdb_record}{args.pdb_suffix}")
        for key in [entry.get("structure_id"), entry.get("protein_id")]:
            if key:
                key = str(key).strip()
                candidates.append(key)
                if "." not in Path(key).name:
                    candidates.append(f"{key}{args.pdb_suffix}")
        seen = set()
        for candidate_name in candidates:
            if not candidate_name or candidate_name in seen:
                continue
            seen.add(candidate_name)
            candidate = pdb_indexes["experimental"].get(candidate_name)
            if candidate is not None:
                return candidate

    if pdb_type == "predicted" and pdb_source_lower == "alphafold" and pdb_record_id:
        if pdb_indexes is None:
            return None
        match = pdb_indexes["alphafold"].get(pdb_record_id)
        if match is not None:
            return match[1]

    if pdb_type == "predicted" and (pdb_source == "" or pdb_source_lower == "nan") and pdb_record_id:
        if pdb_indexes is None:
            return None
        candidate = pdb_indexes["esm3"].get(pdb_record_id)
        if candidate is not None:
            return candidate
    return None


def _build_graph_worker(payload: dict) -> dict:
    graph = build_graph_from_pdb(
        Path(payload["pdb_path"]),
        nneighbor=int(payload["graph_neighbors"]),
        atom_type=str(payload["graph_atom_type"]),
        device="cpu",
    )
    return {
        "structure_key": payload["structure_key"],
        "graph": graph,
    }


def _parse_pdb_worker(payload: dict) -> dict:
    pdb_array = parse_pdb_to_array(
        Path(payload["pdb_path"]),
        nneighbor=int(payload["graph_neighbors"]),
        atom_type=str(payload["graph_atom_type"]),
    )
    return {
        "structure_key": payload["structure_key"],
        "pdb_array": pdb_array,
    }


def _finalize_structure_payload(path: Path, payload: dict, entry: dict, args, legacy, dtype: torch.dtype):
    feature_dims = {}
    if "pst" in args.feature_list:
        legacy_value = resolve_legacy_feature(legacy.get("pst"), entry["protein_id"], entry["structure_id"])
        if legacy_value is None:
            raise RuntimeError(
                "Requested `pst` features but no usable legacy pst.pkl entry was available. "
                "Provide --legacy_feature_dir with pst.pkl or remove pst from --feature_list."
            )
        tensor = _coerce_feature_tensor(legacy_value, dtype=dtype)
        payload["pst"] = tensor
        feature_dims["pst"] = int(tensor.numel())

    if "dssp" in args.feature_list:
        legacy_value = resolve_legacy_feature(legacy.get("dssp"), entry["protein_id"], entry["structure_id"])
        if legacy_value is not None:
            tensor = _coerce_feature_tensor(legacy_value, dtype=dtype)
        else:
            pdb_path = _resolve_pdb_path(entry, args)
            if pdb_path is None:
                raise RuntimeError(
                    f"Requested `dssp` features but no PDB could be resolved for structure `{entry['structure_id']}`."
                )
            tensor = torch.as_tensor(
                compute_dssp_feature(Path(pdb_path), dssp_executable=args.dssp_executable),
                dtype=dtype,
            )
        payload["dssp"] = tensor
        feature_dims["dssp"] = int(tensor.numel())

    _atomic_torch_save(path, payload)
    return feature_dims


def _iter_unique_rows(jobs, column_names):
    tasks = []
    for job in jobs:
        for split_key in ("train_path", "val_path", "test_path"):
            path = Path(job[split_key])
            tasks.append(
                {
                    "path": str(path),
                    "split_group": job["split_group"],
                    "split_name": job["split_name"],
                    "column_names": column_names,
                }
            )

    if not tasks:
        return {}, {}, {}, 0, 0

    unique_sequences = {}
    unique_smiles = {}
    unique_structures = {}
    max_protein_len = 0
    max_smiles_len = 0

    max_workers = min(len(tasks), os.cpu_count() or 1)
    _log(f"Scanning {len(tasks)} split files with up to {max_workers} worker processes", True)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_scan_split_file, payload) for payload in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning split files", unit="file"):
            result = future.result()
            unique_sequences.update(result["unique_sequences"])
            unique_smiles.update(result["unique_smiles"])
            unique_structures.update(result["unique_structures"])
            max_protein_len = max(max_protein_len, result["max_protein_len"])
            max_smiles_len = max(max_smiles_len, result["max_smiles_len"])

    return unique_sequences, unique_smiles, unique_structures, max_protein_len, max_smiles_len


def _resolve_columns(frame, args):
    return {
        "sequence_col": find_first_existing_column(frame, COMMON_SEQUENCE_COLS, explicit=args.sequence_col, required=True),
        "smiles_col": find_first_existing_column(frame, COMMON_SMILES_COLS, explicit=args.smiles_col, required=True),
        "protein_id_col": find_first_existing_column(frame, COMMON_PROTEIN_ID_COLS, explicit=args.protein_id_col, required=False),
        "structure_id_col": find_first_existing_column(frame, COMMON_STRUCTURE_ID_COLS, explicit=args.structure_id_col, required=False),
        "cid_col": find_first_existing_column(frame, COMMON_CID_COLS, explicit=args.cid_col, required=False),
        "pdb_type_col": args.pdb_type_col if args.pdb_type_col and args.pdb_type_col in frame.columns else ("pdb_type" if "pdb_type" in frame.columns else None),
        "pdb_source_col": args.pdb_source_col if args.pdb_source_col and args.pdb_source_col in frame.columns else ("pdb_source" if "pdb_source" in frame.columns else None),
        "pdb_record_col": args.pdb_record_col if args.pdb_record_col and args.pdb_record_col in frame.columns else ("pdbs" if "pdbs" in frame.columns else None),
    }


def _save_protein_payloads(unique_sequences, cache_dir: Path, protein_tokenizer: ProteinSequenceTokenizer, args, legacy):
    dtype = torch.float16 if args.cache_dtype == "float16" else torch.float32
    pending_t5 = []
    feature_dims = {}
    written = 0
    _log(
        f"Protein cache: {len(unique_sequences)} unique sequences | token max_len={args.protein_max_len} | dtype={args.cache_dtype}",
        args.verbose,
    )

    for entry in unique_sequences.values():
        path = protein_cache_path(cache_dir, entry["sequence"], max_len=args.protein_max_len)
        if path.exists() and not args.overwrite:
            continue
        payload = {
            "sequence": entry["sequence"],
            "protein_id": entry["protein_id"],
            "token_ids": torch.tensor(
                protein_tokenizer.encode(entry["sequence"], max_length=args.protein_max_len, add_special_tokens=True),
                dtype=torch.long,
            ),
        }
        if "t5" in args.feature_list:
            legacy_value = resolve_legacy_feature(legacy.get("t5"), entry["protein_id"], entry["sequence"])
            if legacy_value is not None:
                tensor = _coerce_feature_tensor(legacy_value, dtype=dtype)
                payload["t5"] = tensor
                feature_dims["t5"] = int(tensor.numel())
            else:
                pending_t5.append((entry, path, payload))
                continue
        _atomic_torch_save(path, payload)
        written += 1

    if pending_t5:
        device = torch.device(args.device)
        _log(f"ProtT5 phase: loading model on {device}", args.verbose)
        model, tokenizer = load_prot_t5(args.prot_t5_model, device=device)
        sequences = [item[0]["sequence"] for item in pending_t5]
        _log(f"ProtT5 phase: {len(sequences)} sequences require direct computation", args.verbose)
        batches = build_prot_t5_batches(
            sequences,
            max_residues=args.prot_t5_max_residues,
            max_seq_len=args.protein_max_len,
            max_batch=args.prot_t5_max_batch,
        )
        _log(f"ProtT5 phase: {len(batches)} batches", args.verbose)
        batch_lookup = {sequence: (entry, path, payload) for entry, path, payload in pending_t5 for sequence in [entry["sequence"]]}
        for batch in tqdm(batches, desc="Caching ProtT5", unit="batch"):
            embedded = embed_prot_t5_batch(model, tokenizer, batch, device)
            for sequence in batch:
                entry, path, payload = batch_lookup[sequence]
                tensor = torch.as_tensor(embedded[sequence], dtype=dtype)
                payload["t5"] = tensor
                feature_dims["t5"] = int(tensor.numel())
                _atomic_torch_save(path, payload)
                written += 1

    return written, feature_dims


def _save_ligand_payloads(unique_smiles, cache_dir: Path, smiles_tokenizer: RegexSmilesTokenizer, args, legacy):
    dtype = torch.float16 if args.cache_dtype == "float16" else torch.float32
    pending_trfm = []
    pending_lookup = {}
    feature_dims = {}
    written = 0
    _log(
        f"Ligand cache: {len(unique_smiles)} unique SMILES | token max_len={args.smiles_max_len} | dtype={args.cache_dtype}",
        args.verbose,
    )

    for entry in unique_smiles.values():
        path = ligand_cache_path(cache_dir, entry["smiles"])
        if path.exists() and not args.overwrite:
            continue
        payload = {
            "smiles": entry["smiles"],
            "cid": entry.get("cid"),
            "token_ids": torch.tensor(
                smiles_tokenizer.encode(entry["smiles"], max_length=args.smiles_max_len, add_special_tokens=True),
                dtype=torch.long,
            ),
        }
        if "trfm" in args.feature_list:
            legacy_value = None
            if legacy.get("trfm") is not None:
                cid_value = entry.get("cid")
                if cid_value is not None:
                    legacy_value = resolve_legacy_feature(legacy.get("trfm"), cid_value)
                if legacy_value is None:
                    legacy_value = resolve_legacy_feature(legacy.get("trfm"), entry["smiles"])
            if legacy_value is not None:
                tensor = _coerce_feature_tensor(legacy_value, dtype=dtype)
                payload["trfm"] = tensor
                feature_dims["trfm"] = int(tensor.numel())
            else:
                pending_trfm.append((entry, path, payload))
                pending_lookup[entry["smiles"]] = (entry, path, payload)
        if "molformer" in args.feature_list:
            legacy_value = None
            if legacy.get("molformer") is not None:
                cid_value = entry.get("cid")
                if cid_value is not None:
                    legacy_value = resolve_legacy_feature(legacy.get("molformer"), cid_value)
                if legacy_value is None:
                    legacy_value = resolve_legacy_feature(legacy.get("molformer"), entry["smiles"])
            if legacy_value is None:
                raise RuntimeError(
                    "Requested `molformer` features but no usable legacy molformer.pkl entry was available. "
                    "Provide --legacy_feature_dir with molformer.pkl or remove molformer from --feature_list."
                )
            tensor = _coerce_feature_tensor(legacy_value, dtype=dtype)
            payload["molformer"] = tensor
            feature_dims["molformer"] = int(tensor.numel())
        if "trfm" not in args.feature_list or "trfm" in payload:
            _atomic_torch_save(path, payload)
            written += 1

    if pending_trfm:
        device = torch.device(args.device)
        _log(f"SMILES Transformer phase: loading model on {device}", args.verbose)
        model, vocab = load_smiles_transformer(
            weights_path=args.trfm_weights_path,
            vocab_path=args.trfm_vocab_path,
            device=device,
        )
        smiles_values = [entry["smiles"] for entry, _, _ in pending_trfm]
        _log(f"SMILES Transformer phase: {len(smiles_values)} SMILES require direct computation", args.verbose)
        for start in tqdm(range(0, len(smiles_values), args.trfm_batch_size), desc="Caching TRFM", unit="batch"):
            batch_smiles = smiles_values[start : start + args.trfm_batch_size]
            embedded = embed_smiles_trfm_batch(batch_smiles, model=model, vocab=vocab, device=device)
            for smiles in batch_smiles:
                entry, path, payload = pending_lookup[smiles]
                tensor = torch.as_tensor(embedded[smiles], dtype=dtype)
                payload["trfm"] = tensor
                feature_dims["trfm"] = int(tensor.numel())
                _atomic_torch_save(path, payload)
                written += 1

    return written, feature_dims


def _save_structure_payloads(unique_structures, cache_dir: Path, args, legacy):
    dtype = torch.float16 if args.cache_dtype == "float16" else torch.float32
    feature_dims = {}
    written = 0
    skipped = 0
    graph_device = str(args.graph_device)
    use_gpu_graphs = graph_device.startswith("cuda")
    existing_structure_cache = set() if args.overwrite else _build_existing_structure_cache_index(cache_dir)
    _log(f"Structure cache: {len(unique_structures)} unique PDB structures", args.verbose)
    graph_tasks = []
    for entry in tqdm(unique_structures.values(), desc="Resolving structures", unit="structure"):
        path = structure_cache_path(cache_dir, structure_id=entry["structure_id"], fallback_sequence=entry["sequence"])
        if not args.overwrite and str(path) in existing_structure_cache:
            continue
        payload = {
            "structure_id": entry["structure_id"],
            "protein_id": entry["protein_id"],
        }

        legacy_graph = resolve_legacy_feature(legacy.get("graph"), entry["protein_id"], entry["structure_id"])
        if legacy_graph is not None:
            payload["graph"] = legacy_graph
            current_dims = _finalize_structure_payload(path, payload, entry, args, legacy, dtype)
            feature_dims.update(current_dims)
            written += 1
        else:
            pdb_path = _resolve_pdb_path(entry, args)
            if pdb_path is None:
                _log(
                    f"Skipping structure cache for `{entry['structure_id']}` because no matching PDB file was found.",
                    args.verbose,
                )
                skipped += 1
                continue
            graph_tasks.append(
                {
                    "structure_key": entry["structure_id"],
                    "pdb_path": str(pdb_path),
                    "graph_neighbors": args.graph_neighbors,
                    "graph_atom_type": args.graph_atom_type,
                    "path": path,
                    "payload": payload,
                    "entry": entry,
                }
            )

    if graph_tasks:
        if use_gpu_graphs:
            parse_workers = max(1, int(args.graph_parse_workers))
            prefetch_limit = max(1, int(args.graph_prefetch))
            _log(
                f"Building {len(graph_tasks)} structure graphs on {graph_device} with {parse_workers} CPU parse threads and prefetch={prefetch_limit}",
                args.verbose,
            )
            task_iter = iter(graph_tasks)
            pending = {}
            with ThreadPoolExecutor(max_workers=parse_workers) as pool:
                def submit_until_full():
                    while len(pending) < prefetch_limit:
                        try:
                            task = next(task_iter)
                        except StopIteration:
                            break
                        future = pool.submit(_parse_pdb_worker, task)
                        pending[future] = task

                submit_until_full()
                with tqdm(total=len(graph_tasks), desc="Building structure graphs", unit="structure") as progress:
                    while pending:
                        future = next(as_completed(list(pending.keys())))
                        task = pending.pop(future)
                        try:
                            result = future.result()
                            graph = build_graph_from_array(
                                result["pdb_array"],
                                nneighbor=args.graph_neighbors,
                                device=graph_device,
                            )
                            task["payload"]["graph"] = graph
                            current_dims = _finalize_structure_payload(task["path"], task["payload"], task["entry"], args, legacy, dtype)
                            feature_dims.update(current_dims)
                            written += 1
                            del graph
                            del result
                            if torch.cuda.is_available() and graph_device.startswith("cuda"):
                                torch.cuda.empty_cache()
                        except Exception as exc:
                            skipped += 1
                            print(f"Skipping structure `{task['entry']['structure_id']}` because graph build failed for {task['pdb_path']}: {exc}", flush=True)
                        gc.collect()
                        progress.update(1)
                        submit_until_full()
        else:
            max_workers = min(len(graph_tasks), os.cpu_count() or 1)
            _log(f"Building {len(graph_tasks)} structure graphs with up to {max_workers} CPU workers", args.verbose)
            futures = {}
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                for task in graph_tasks:
                    future = pool.submit(_build_graph_worker, task)
                    futures[future] = task
                for future in tqdm(as_completed(futures), total=len(futures), desc="Building structure graphs", unit="structure"):
                    task = futures[future]
                    try:
                        result = future.result()
                        task["payload"]["graph"] = result["graph"]
                        current_dims = _finalize_structure_payload(task["path"], task["payload"], task["entry"], args, legacy, dtype)
                        feature_dims.update(current_dims)
                        written += 1
                    except Exception as exc:
                        skipped += 1
                        print(f"Skipping structure `{task['entry']['structure_id']}` because graph build failed for {task['pdb_path']}: {exc}", flush=True)
    return written, feature_dims, skipped


def main():
    parser = argparse.ArgumentParser(description="Build reusable DEKP caches for EMULaToR split retraining.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--cache_dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--dataset_df_path", type=str, default=str(DEFAULT_DATASET_DF_PATH))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--feature_list", type=str, default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--sequence_col", type=str, default=None)
    parser.add_argument("--smiles_col", type=str, default=None)
    parser.add_argument("--protein_id_col", type=str, default=None)
    parser.add_argument("--structure_id_col", type=str, default=None)
    parser.add_argument("--cid_col", type=str, default=None)
    parser.add_argument("--pdb_type_col", type=str, default=None)
    parser.add_argument("--pdb_source_col", type=str, default=None)
    parser.add_argument("--pdb_record_col", type=str, default=None)
    parser.add_argument("--legacy_feature_dir", type=str, default=None)
    parser.add_argument("--experimental_pdb_dir", type=str, default=None)
    parser.add_argument("--alphafold_pdb_dir", type=str, default=None)
    parser.add_argument("--esm3_pdb_dir", type=str, default=None)
    parser.add_argument("--pdb_suffix", type=str, default=".pdb")
    parser.add_argument("--prot_t5_model", type=str, default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--trfm_weights_path", type=str, default=None)
    parser.add_argument("--trfm_vocab_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--graph_device", type=str, default=None)
    parser.add_argument("--cache_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--protein_max_len", type=int, default=2500)
    parser.add_argument("--smiles_max_len", type=int, default=512)
    parser.add_argument("--prot_t5_max_residues", type=int, default=4000)
    parser.add_argument("--prot_t5_max_batch", type=int, default=8)
    parser.add_argument("--trfm_batch_size", type=int, default=256)
    parser.add_argument("--graph_neighbors", type=int, default=32)
    parser.add_argument("--graph_atom_type", type=str, default="CA")
    parser.add_argument("--graph_parse_workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--graph_prefetch", type=int, default=8)
    parser.add_argument("--dssp_executable", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.verbose = True

    args.base_dir = Path(args.base_dir).expanduser()
    args.cache_dir = Path(args.cache_dir).expanduser()
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    args.feature_list = [item.strip() for item in args.feature_list.split(",") if item.strip()]
    args.graph_device = args.graph_device or args.device
    if str(args.graph_device).startswith("cuda") and not torch.cuda.is_available():
        _log(f"Requested graph_device={args.graph_device} but CUDA is unavailable; falling back to cpu", True)
        args.graph_device = "cpu"
    args._pdb_indexes = _build_pdb_indexes(args)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_df_path = Path(args.dataset_df_path).expanduser()
    jobs = []
    use_full_dataset = dataset_df_path.exists()
    if use_full_dataset:
        _log(f"Loading full dataset dataframe from {dataset_df_path}", args.verbose)
        frame = pd.read_parquet(dataset_df_path, columns=["sequence", "smiles", "pdbs", "pdb_type", "pdb_source"])
        _log(f"Loaded full frame with {len(frame)} rows and columns={frame.columns.tolist()}", args.verbose)
        column_names = {
            "sequence_col": "sequence",
            "smiles_col": "smiles",
            "protein_id_col": "sequence",
            "structure_id_col": "pdbs" if "pdbs" in frame.columns else "sequence",
            "cid_col": None,
            "pdb_type_col": "pdb_type" if "pdb_type" in frame.columns else None,
            "pdb_source_col": "pdb_source" if "pdb_source" in frame.columns else None,
            "pdb_record_col": "pdbs" if "pdbs" in frame.columns else None,
        }
    else:
        jobs = discover_split_jobs(args.base_dir, split_groups=args.split_groups, thresholds=args.thresholds)
        if not jobs:
            raise FileNotFoundError(f"No split jobs discovered in {args.base_dir} and no dataset_df_path found at {dataset_df_path}")
        _log(f"Discovered {len(jobs)} split jobs in {args.base_dir} for groups={args.split_groups} and thresholds={args.thresholds}", args.verbose)
        sample_frame = read_table(Path(jobs[0]["train_path"]))
        _log(f"Sample frame columns: {sample_frame.columns.tolist()}", args.verbose)
        column_names = _resolve_columns(sample_frame, args)
        if column_names["protein_id_col"] is None:
            column_names["protein_id_col"] = column_names["sequence_col"]
        if column_names["structure_id_col"] is None:
            column_names["structure_id_col"] = column_names["protein_id_col"]
        _log(f"Resolved column names: {column_names}", args.verbose)

    started = time.time()
    if use_full_dataset:
        unique_sequences, unique_smiles, unique_structures, max_protein_len, max_smiles_len = _iter_unique_rows_from_frame(frame, column_names)
    else:
        unique_sequences, unique_smiles, unique_structures, max_protein_len, max_smiles_len = _iter_unique_rows(jobs, column_names)
    args.protein_max_len = max(args.protein_max_len, max_protein_len)
    args.smiles_max_len = max(args.smiles_max_len, max_smiles_len)
    _log(
        f"Discovered {len(jobs) if jobs else 1} data source(s) | {len(unique_sequences)} sequences | {len(unique_smiles)} smiles | {len(unique_structures)} structures",
        args.verbose,
    )
    _log(f"CUDA available={torch.cuda.is_available()} | selected device={args.device}", args.verbose)
    if args.verbose and torch.cuda.is_available() and str(args.device).startswith("cuda"):
        try:
            device_index = int(str(args.device).split(":")[-1]) if ":" in str(args.device) else torch.cuda.current_device()
            _log(
                f"CUDA device {device_index}: {torch.cuda.get_device_name(device_index)} | capability={torch.cuda.get_device_capability(device_index)}",
                True,
            )
        except Exception:
            pass

    protein_tokenizer = ProteinSequenceTokenizer()
    smiles_tokenizer = RegexSmilesTokenizer.from_smiles(unique_smiles.keys())
    protein_tokenizer.save(protein_vocab_path(args.cache_dir))
    smiles_tokenizer.save(smiles_vocab_path(args.cache_dir))

    legacy = {}
    if args.legacy_feature_dir:
        legacy = load_legacy_feature_pickles(Path(args.legacy_feature_dir), set(args.feature_list) | {"graph"})

    protein_written, protein_dims = _save_protein_payloads(unique_sequences, args.cache_dir, protein_tokenizer, args, legacy)
    ligand_written, ligand_dims = _save_ligand_payloads(unique_smiles, args.cache_dir, smiles_tokenizer, args, legacy)
    structure_written, structure_dims, structure_skipped = _save_structure_payloads(unique_structures, args.cache_dir, args, legacy)

    feature_dims = {}
    feature_dims.update(protein_dims)
    feature_dims.update(ligand_dims)
    feature_dims.update(structure_dims)

    manifest = {
        "cache_version": 1,
        "base_dir": str(args.base_dir),
        "cache_dir": str(args.cache_dir),
        "feature_list": args.feature_list,
        "resolved_columns": column_names,
        "protein_max_len": int(args.protein_max_len),
        "smiles_max_len": int(args.smiles_max_len),
        "protein_vocab_size": len(protein_tokenizer),
        "smiles_vocab_size": len(smiles_tokenizer),
        "protein_pad_id": protein_tokenizer.pad_id,
        "smiles_pad_id": smiles_tokenizer.pad_id,
        "feature_dims": feature_dims,
        "counts": {
            "split_jobs": len(jobs),
            "unique_sequences": len(unique_sequences),
            "unique_smiles": len(unique_smiles),
            "unique_structures": len(unique_structures),
            "proteins_written": protein_written,
            "ligands_written": ligand_written,
            "structures_written": structure_written,
            "structures_skipped": structure_skipped,
        },
        "legacy_feature_dir": str(args.legacy_feature_dir) if args.legacy_feature_dir else None,
        "prot_t5_model": args.prot_t5_model if "t5" in args.feature_list else None,
        "trfm_weights_path": str(args.trfm_weights_path) if args.trfm_weights_path else None,
        "trfm_vocab_path": str(args.trfm_vocab_path) if args.trfm_vocab_path else None,
        "elapsed_seconds": time.time() - started,
    }
    save_json(args.cache_dir / "manifest.json", manifest)

    _log(f"Saved cache manifest to {args.cache_dir / 'manifest.json'}", True)
    _log(
        f"Done: proteins_written={protein_written} | ligands_written={ligand_written} | structures_written={structure_written} | structures_skipped={structure_skipped} | elapsed={manifest['elapsed_seconds']:.1f}s",
        True,
    )


if __name__ == "__main__":
    main()
