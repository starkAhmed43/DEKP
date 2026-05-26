import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import optuna
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_FEATURES,
    DEFAULT_SPLIT_GROUPS,
    apply_manifest_paths_to_jobs,
    discover_split_jobs,
    normalize_threshold_args,
    resolve_metric_value,
)
from emulator_bench.run_split_benchmarks import maybe_cache_embeddings


TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def _metric_direction(metric: str) -> str:
    return "minimize" if metric in {"rmse", "mse", "mae", "loss"} else "maximize"


def _sqlite_path_from_storage(storage: str | None):
    if not storage or not storage.startswith("sqlite:///"):
        return None
    parsed = urlparse(storage)
    if parsed.scheme != "sqlite":
        return None
    raw_path = unquote(parsed.path or "")
    return Path(raw_path) if raw_path else None


def _sqlite_has_optuna_schema(db_path: Path) -> bool:
    with sqlite3.connect(db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    return "version_info" in tables


def prepare_optuna_storage(args) -> None:
    db_path = _sqlite_path_from_storage(args.storage)
    if db_path is None:
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        return
    if args.reset_storage:
        db_path.unlink()
        print(f"Removed existing Optuna storage: {db_path}")
        return
    if not _sqlite_has_optuna_schema(db_path):
        raise RuntimeError(
            f"Optuna storage exists but does not contain a valid Optuna schema: {db_path}. "
            "Use a new --storage path or rerun with --reset_storage."
        )


def suggest_hparams(trial: optuna.Trial, args) -> dict:
    batch_size = int(args.batch_size) if args.batch_size is not None else trial.suggest_categorical("batch_size", [64, 96, 128, 192, 256])
    return {
        "batch_size": batch_size,
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "dropout": float(args.dropout),
    }


def run_trial_job(job: dict, seed: int, hparams: dict, args, trial_number: int) -> float:
    trial_root = Path(job["root_dir"]) / "dekp_optuna_runs" / f"trial_{trial_number}" / f"seed_{seed}"
    trial_root.mkdir(parents=True, exist_ok=True)
    metrics_file = trial_root / f"final_results_{args.eval_split}.csv"
    if not metrics_file.exists() or args.overwrite_runs:
        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--train_path",
            job["train_path"],
            "--val_path",
            job["val_path"],
            "--test_path",
            job["test_path"],
            "--cache_dir",
            args.cache_dir,
            "--out_dir",
            str(trial_root),
            "--task_name",
            f"optuna_{trial_number}_{job['split_group']}_{job['split_name']}_seed{seed}",
            "--feature_list",
            args.feature_list,
            "--batch_size",
            str(hparams["batch_size"]),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(hparams["lr"]),
            "--weight_decay",
            str(hparams["weight_decay"]),
            "--hidden",
            str(args.hidden),
            "--num_layers",
            str(args.num_layers),
            "--kernel_size",
            str(args.kernel_size),
            "--dropout",
            str(hparams["dropout"]),
            "--device",
            args.device,
            "--num_workers",
            str(args.num_workers),
            "--prefetch_factor",
            str(args.prefetch_factor),
            "--cache_items",
            str(args.cache_items),
            "--seed",
            str(seed),
        ]
        for key in ["sequence_col", "smiles_col", "protein_id_col", "structure_id_col", "target_col"]:
            value = getattr(args, key)
            if value:
                cmd.extend([f"--{key}", value])
        if args.persistent_workers:
            cmd.append("--persistent_workers")
        if args.pin_memory:
            cmd.append("--pin_memory")
        if args.preload_proteins:
            cmd.append("--preload_proteins")
        if args.preload_ligands:
            cmd.append("--preload_ligands")
        if args.preload_structures:
            cmd.append("--preload_structures")
        if args.compile_model:
            cmd.append("--compile_model")
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    metrics = pd.read_csv(metrics_file).iloc[0].to_dict()
    return resolve_metric_value(metrics, args.metric)


def main():
    parser = argparse.ArgumentParser(description="Tune DEKP optimizer hyperparameters with Optuna.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--cache_dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--manifests_dir", type=str, default=None)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--feature_list", type=str, default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--sequence_col", type=str, default=None)
    parser.add_argument("--smiles_col", type=str, default=None)
    parser.add_argument("--protein_id_col", type=str, default=None)
    parser.add_argument("--structure_id_col", type=str, default=None)
    parser.add_argument("--target_col", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=9)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--preload_proteins", action="store_true")
    parser.add_argument("--preload_ligands", action="store_true")
    parser.add_argument("--preload_structures", action="store_true")
    parser.add_argument("--cache_items", type=int, default=512)
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--metric", choices=["rmse", "mse", "mae", "r2", "pearson", "spearman", "loss"], default="rmse")
    parser.add_argument("--eval_split", choices=["val", "test"], default="val")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407])
    parser.add_argument("--overwrite_runs", action="store_true")
    parser.add_argument("--study_name", type=str, default="dekp_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--reset_storage", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--legacy_feature_dir", type=str, default=None)
    parser.add_argument("--experimental_pdb_dir", type=str, default=None)
    parser.add_argument("--alphafold_pdb_dir", type=str, default=None)
    parser.add_argument("--esm3_pdb_dir", type=str, default=None)
    parser.add_argument("--protein_max_len", type=int, default=2500)
    parser.add_argument("--smiles_max_len", type=int, default=512)
    parser.add_argument("--prot_t5_model", type=str, default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--prot_t5_max_residues", type=int, default=4000)
    parser.add_argument("--prot_t5_max_batch", type=int, default=8)
    parser.add_argument("--trfm_weights_path", type=str, default=None)
    parser.add_argument("--trfm_vocab_path", type=str, default=None)
    parser.add_argument("--trfm_batch_size", type=int, default=256)
    parser.add_argument("--graph_neighbors", type=int, default=32)
    parser.add_argument("--graph_atom_type", type=str, default="CA")
    parser.add_argument("--cache_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--cache_device", type=str, default="cuda:0")
    args = parser.parse_args()

    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    maybe_cache_embeddings(args)
    prepare_optuna_storage(args)
    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {args.base_dir}")
    if args.manifests_dir:
        jobs = apply_manifest_paths_to_jobs(jobs, Path(args.manifests_dir).expanduser(), require=True)

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    study_kwargs = {
        "direction": _metric_direction(args.metric),
        "study_name": args.study_name,
        "sampler": sampler,
    }
    if args.storage:
        study_kwargs["storage"] = args.storage
        study_kwargs["load_if_exists"] = True
    study = optuna.create_study(**study_kwargs)

    trials_rows = []

    def objective(trial: optuna.Trial):
        hparams = suggest_hparams(trial, args)
        values = []
        for job in jobs:
            for seed in args.seeds:
                metric_value = run_trial_job(job, seed, hparams, args, trial.number)
                values.append(metric_value)
        objective_value = float(sum(values) / len(values))
        trials_rows.append({"trial_number": int(trial.number), "objective": objective_value, **hparams})
        return objective_value

    study.optimize(objective, n_trials=args.n_trials)

    out_dir = Path(args.base_dir) / "optuna_studies"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trials_rows).to_csv(out_dir / f"{args.study_name}_trials.csv", index=False)
    best_payload = dict(study.best_params)
    best_payload.update(
        {
            "study_name": args.study_name,
            "best_trial_number": int(study.best_trial.number),
            "best_value": float(study.best_value),
            "direction": _metric_direction(args.metric),
            "metric": args.metric,
            "feature_list": args.feature_list,
        }
    )
    with open(out_dir / f"{args.study_name}_best_hparams.json", "w", encoding="utf-8") as handle:
        json.dump(best_payload, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
