import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_MANIFESTS_DIR,
    DEFAULT_SPLIT_GROUPS,
    KEY_COLUMNS,
    atomic_json,
    discover_split_jobs,
    load_json,
    manifest_path,
    normalize_sequence,
    normalize_threshold_args,
    read_table,
    require_columns,
    stable_hash,
    write_table,
)


DEFAULT_EXPERIMENTAL_PDB_DIR = Path("/home/da24s023/github/EMULaToR/data/intermediate/processed_exp_pdb")
DEFAULT_ALPHAFOLD_PDB_DIR = Path("/home/da24s023/github/EMULaToR/data/intermediate/alphafold")
DEFAULT_ESM_PDB_DIR = Path("/home/da24s023/github/EMULaToR/data/intermediate/esm")
PDB_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9])([0-9][A-Za-z0-9]{3})(?![A-Za-z0-9])")
STRUCTURE_MATCH_COLUMNS = ["smiles", "sequence", "value", "smiles_hash", "log10_value"]


def _norm_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "nat"} else text


def _extract_pdb_candidates(value) -> List[str]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        raw_items = [str(item) for item in value]
    else:
        text = _norm_text(value)
        if not text:
            return []
        if text[0] in "[(" and text[-1] in "])":
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = None
            raw_items = [str(item) for item in parsed] if isinstance(parsed, (list, tuple, set)) else re.split(r"[,\s;]+", text)
        else:
            raw_items = re.split(r"[,\s;]+", text)

    out, seen = [], set()
    for item in raw_items:
        token = Path(_norm_text(item)).stem
        if not token:
            continue
        if "|" in token:
            if token not in seen:
                seen.add(token)
                out.append(token)
            continue
        upper = token.upper()
        if re.fullmatch(r"[0-9A-Z]{4}", upper):
            if upper not in seen:
                seen.add(upper)
                out.append(upper)
            continue
        for match in PDB_ID_PATTERN.findall(upper):
            if match not in seen:
                seen.add(match)
                out.append(match)
    return out


def _build_experimental_index(directory: Path) -> Dict[str, str]:
    directory = Path(directory).expanduser()
    return {path.stem.upper(): str(path) for path in directory.glob("*.pdb")} if directory.exists() else {}


def _build_alphafold_index(directory: Path) -> Dict[str, str]:
    directory = Path(directory).expanduser()
    if not directory.exists():
        return {}
    index: Dict[str, Tuple[int, str]] = {}
    for path in directory.glob("AF-*-F1-model_v*.pdb"):
        stem = path.stem
        accession, _, version_text = stem[len("AF-") :].partition("-F1-model_v")
        try:
            version = int(version_text)
        except ValueError:
            version = -1
        current = index.get(accession.upper())
        if current is None or version > current[0]:
            index[accession.upper()] = (version, str(path))
    return {key: path for key, (_version, path) in index.items()}


def _build_esm_index(directory: Path) -> Dict[str, str]:
    directory = Path(directory).expanduser()
    if not directory.exists():
        return {}
    index = {}
    prefix = "ESM3-open-small-"
    for path in directory.glob(f"{prefix}*.pdb"):
        key = path.name[len(prefix) : -len(".pdb")]
        index[key] = str(path)
        index[key.upper()] = str(path)
    return index


def _extract_pdb_sequences(path: str) -> List[dict]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(Path(path).stem, path)
    entries = []
    for model in structure:
        for chain in model:
            residues = []
            for residue in chain:
                if is_aa(residue, standard=False):
                    residues.append(seq1(residue.get_resname(), custom_map={"MSE": "M"}, undef_code="X"))
            if residues:
                entries.append({"chain_id": str(chain.id), "sequence": "".join(residues)})
        break
    return entries


def _make_aligner():
    aligner = PairwiseAligner(mode="global")
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = 0.0
    aligner.extend_gap_score = 0.0
    return aligner


def _identity_pct(aligner, query: str, subject: str) -> float:
    if not query or not subject:
        return 0.0
    return 100.0 * float(aligner.score(query, subject)) / max(len(query), len(subject))


def _row_key(row: dict) -> str:
    payload = {column: str(row.get(column, "")) for column in STRUCTURE_MATCH_COLUMNS}
    return stable_hash(json.dumps(payload, sort_keys=True))


def _candidate_dataset_path(base_dir: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Missing candidate parquet: {path}")
        return path

    preferred = [
        base_dir / "kcat_kinetic_params_3d_aligned.parquet",
        base_dir / "kcat_kinetic_params_3d.parquet",
    ]
    for path in preferred:
        if path.exists():
            return path

    candidates = sorted(base_dir.glob("*_3d_aligned.parquet")) + sorted(base_dir.glob("*_3d.parquet"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No 3D candidate parquet found in {base_dir}")


def _candidate_lookup(base_dir: Path, candidate_df_path: Path | None = None) -> Dict[str, List[dict]]:
    path = _candidate_dataset_path(base_dir, candidate_df_path)
    frame = read_table(path)
    require_columns(frame, KEY_COLUMNS + ["pdbs", "pdb_source", "pdb_type"], path)
    lookup: Dict[str, List[dict]] = {}
    columns = KEY_COLUMNS + ["pdbs", "pdb_source", "pdb_type"]
    for row in frame[columns].to_dict("records"):
        lookup.setdefault(_row_key(row), []).append(row)
        seq_key = "sequence:" + stable_hash(normalize_sequence(row.get("sequence", "")))
        lookup.setdefault(seq_key, []).append(row)
    return lookup


class StructureResolver:
    def __init__(self, args):
        self.experimental_index = _build_experimental_index(args.experimental_pdb_dir)
        self.alphafold_index = _build_alphafold_index(args.alphafold_pdb_dir)
        self.esm_index = _build_esm_index(args.esm_pdb_dir)
        self.identity_threshold = float(args.identity_threshold)
        self.aligner = _make_aligner()
        cache_dir = Path(args.base_dir) / "_structure_alignment_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.sequence_cache_path = cache_dir / "pdb_sequence_cache.json"
        self.selection_cache_path = cache_dir / "selection_cache.json"
        self.sequence_cache = load_json(self.sequence_cache_path, {})
        self.selection_cache = load_json(self.selection_cache_path, {})

    def save(self):
        atomic_json(self.sequence_cache_path, self.sequence_cache)
        atomic_json(self.selection_cache_path, self.selection_cache)

    def _pdb_sequences(self, pdb_id: str) -> List[dict]:
        if pdb_id in self.sequence_cache:
            return self.sequence_cache[pdb_id]
        path = self.experimental_index[pdb_id]
        try:
            entries = _extract_pdb_sequences(path)
        except Exception as exc:
            entries = []
            self.sequence_cache[f"{pdb_id}__error"] = str(exc)
        self.sequence_cache[pdb_id] = entries
        return entries

    def resolve(self, row: dict, candidates: List[dict]) -> dict:
        task_key = stable_hash(
            json.dumps(
                {
                    "sequence": normalize_sequence(row.get("sequence", "")),
                    "candidates": [
                        [_norm_text(item.get("pdbs")), _norm_text(item.get("pdb_source")), _norm_text(item.get("pdb_type"))]
                        for item in candidates
                    ],
                },
                sort_keys=True,
            )
        )
        if task_key in self.selection_cache:
            return dict(self.selection_cache[task_key])

        sequence = normalize_sequence(row.get("sequence", ""))
        exp_candidates, alpha_candidates, esm_candidates = [], [], []
        for item in candidates:
            pdbs = _norm_text(item.get("pdbs"))
            source = _norm_text(item.get("pdb_source")).lower()
            kind = _norm_text(item.get("pdb_type")).lower()
            if kind == "experimental" or source == "pdbe":
                exp_candidates.extend(pdb for pdb in _extract_pdb_candidates(pdbs) if pdb.upper() in self.experimental_index)
            elif kind == "predicted" and source == "alphafold":
                accession = pdbs.split("|", 1)[0].upper()
                if accession in self.alphafold_index:
                    alpha_candidates.append(accession)
            else:
                key = pdbs if pdbs in self.esm_index else pdbs.upper()
                if key in self.esm_index:
                    esm_candidates.append(key)

        best_pdb, best_chain, best_identity = None, "", -1.0
        for pdb_id in sorted(set(p.upper() for p in exp_candidates)):
            for entry in self._pdb_sequences(pdb_id):
                identity = _identity_pct(self.aligner, sequence, normalize_sequence(entry.get("sequence", "")))
                if identity > best_identity:
                    best_pdb, best_chain, best_identity = pdb_id, str(entry.get("chain_id", "")), identity

        if best_pdb and best_identity >= self.identity_threshold:
            result = {
                "structure_path": self.experimental_index[best_pdb],
                "pdbs": best_pdb,
                "pdb_source": "PDBe",
                "pdb_type": "experimental",
                "chain_id": best_chain or "A",
                "resolved_structure_status": "selected_experimental",
                "resolved_structure_identity": round(best_identity, 4),
            }
        elif alpha_candidates:
            accession = sorted(set(alpha_candidates))[0]
            result = {
                "structure_path": self.alphafold_index[accession],
                "pdbs": accession,
                "pdb_source": "AlphaFold",
                "pdb_type": "predicted",
                "chain_id": "A",
                "resolved_structure_status": "selected_alphafold",
                "resolved_structure_identity": None if best_identity < 0 else round(best_identity, 4),
            }
        elif esm_candidates:
            key = sorted(set(esm_candidates))[0]
            result = {
                "structure_path": self.esm_index[key],
                "pdbs": key,
                "pdb_source": "ESM",
                "pdb_type": "predicted",
                "chain_id": "A",
                "resolved_structure_status": "selected_esm",
                "resolved_structure_identity": None if best_identity < 0 else round(best_identity, 4),
            }
        else:
            result = {
                "structure_path": "",
                "pdbs": "",
                "pdb_source": "",
                "pdb_type": "",
                "chain_id": "",
                "resolved_structure_status": "unresolved",
                "resolved_structure_identity": None if best_identity < 0 else round(best_identity, 4),
            }
        self.selection_cache[task_key] = result
        return dict(result)


def _sample_id(row: dict) -> str:
    return stable_hash(json.dumps({column: str(row.get(column, "")) for column in KEY_COLUMNS}, sort_keys=True))


def prepare_one_split(path: Path, output_path: Path, candidates: Dict[str, List[dict]], resolver: StructureResolver, args) -> dict:
    frame = read_table(path)
    require_columns(frame, KEY_COLUMNS, path)
    if args.limit_rows:
        frame = frame.head(args.limit_rows).copy()
    rows = []
    failed = 0
    for row in tqdm(frame.to_dict("records"), desc=f"Preparing {path.parent.name}/{path.name}", unit="row"):
        exact_candidates = candidates.get(_row_key(row), [])
        sequence_candidates = candidates.get("sequence:" + stable_hash(normalize_sequence(row.get("sequence", ""))), [])
        candidate_groups = [exact_candidates]
        if sequence_candidates and sequence_candidates != exact_candidates:
            candidate_groups.append(sequence_candidates)

        result = {}
        for row_candidates in candidate_groups:
            if not row_candidates:
                continue
            result = resolver.resolve(row, row_candidates)
            if result.get("structure_path"):
                break
        if not result.get("structure_path"):
            failed += 1
            if args.fail_on_unresolved:
                raise RuntimeError(f"Could not resolve structure for row sample_id={_sample_id(row)}")
            continue
        out = dict(row)
        out["sample_id"] = _sample_id(row)
        out["target"] = float(row[args.target_col])
        out.update(result)
        rows.append(out)
    out_frame = pd.DataFrame(rows)
    write_table(output_path, out_frame)
    return {
        "input_path": str(path),
        "output_path": str(output_path),
        "rows_in": int(len(frame)),
        "rows_out": int(len(out_frame)),
        "rows_unresolved": int(failed),
    }


def prepared_manifest_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        frame = read_table(path)
    except Exception:
        return False
    required = {"sequence", "smiles", "structure_path", "chain_id", "target", "pdbs", "pdb_source", "pdb_type"}
    return bool(required.issubset(frame.columns))


def main():
    parser = argparse.ArgumentParser(description="Prepare DEKP EMULaToR split manifests with one resolved structure per row.")
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--manifests_dir", type=Path, default=DEFAULT_MANIFESTS_DIR)
    parser.add_argument("--candidate_df_path", type=Path, default=None)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--identity_threshold", type=float, default=90.0)
    parser.add_argument("--experimental_pdb_dir", type=Path, default=DEFAULT_EXPERIMENTAL_PDB_DIR)
    parser.add_argument("--alphafold_pdb_dir", type=Path, default=DEFAULT_ALPHAFOLD_PDB_DIR)
    parser.add_argument("--esm_pdb_dir", type=Path, default=DEFAULT_ESM_PDB_DIR)
    parser.add_argument("--limit_rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail_on_unresolved", action="store_true")
    args = parser.parse_args()

    args.base_dir = Path(args.base_dir).expanduser()
    args.manifests_dir = Path(args.manifests_dir).expanduser()
    args.experimental_pdb_dir = Path(args.experimental_pdb_dir).expanduser()
    args.alphafold_pdb_dir = Path(args.alphafold_pdb_dir).expanduser()
    args.esm_pdb_dir = Path(args.esm_pdb_dir).expanduser()
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)

    jobs = discover_split_jobs(args.base_dir, args.split_groups, args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")
    candidates = _candidate_lookup(args.base_dir, args.candidate_df_path)
    resolver = StructureResolver(args)

    summaries = []
    for job in jobs:
        for split in ("train", "val", "test"):
            out_path = manifest_path(args.manifests_dir, job, split)
            if not args.overwrite and prepared_manifest_is_valid(out_path):
                summaries.append({"output_path": str(out_path), "status": "skipped_exists"})
                continue
            summaries.append(prepare_one_split(Path(job[f"{split}_path"]), out_path, candidates, resolver, args))
            resolver.save()

    resolver.save()
    summary = {"jobs": len(jobs), "splits": summaries}
    atomic_json(args.manifests_dir / "prepare_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
