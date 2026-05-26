import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import optuna

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import normalize_threshold_args
from emulator_bench.tune_optuna import _metric_direction, prepare_optuna_storage
from emulator_bench.run_split_benchmarks import maybe_cache_embeddings


TUNE_SCRIPT = REPO_ROOT / "emulator_bench" / "tune_optuna.py"


def _split_trials(total_trials: int, num_workers: int):
    base = total_trials // num_workers
    remainder = total_trials % num_workers
    return [base + (1 if idx < remainder else 0) for idx in range(num_workers)]


def main():
    parser = argparse.ArgumentParser(description="Launch multiple single-GPU Optuna workers against a shared study.")
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--manifests_dir", type=str, default=None)
    parser.add_argument("--split_groups", nargs="+", default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--feature_list", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
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
    parser.add_argument("--metric", default="rmse")
    parser.add_argument("--eval_split", default="val")
    parser.add_argument("--n_trials", type=int, required=True)
    parser.add_argument("--study_name", type=str, default="dekp_optuna")
    parser.add_argument("--storage", type=str, required=True)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--reset_storage", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--overwrite_runs", action="store_true")
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
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=9)
    parser.add_argument("--dropout", type=float, default=0.5)
    args = parser.parse_args()

    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    maybe_cache_embeddings(args)
    prepare_optuna_storage(args)
    optuna.create_study(
        direction=_metric_direction(args.metric),
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
    )

    trial_splits = _split_trials(args.n_trials, len(args.gpus))
    procs = []
    try:
        for worker_index, (gpu_id, worker_trials) in enumerate(zip(args.gpus, trial_splits)):
            if worker_trials <= 0:
                continue
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            cmd = [
                sys.executable,
                str(TUNE_SCRIPT),
                "--base_dir",
                args.base_dir,
                "--cache_dir",
                args.cache_dir,
                "--feature_list",
                args.feature_list,
                "--epochs",
                str(args.epochs),
                "--device",
                "cuda:0",
                "--num_workers",
                str(args.num_workers),
                "--prefetch_factor",
                str(args.prefetch_factor),
                "--cache_items",
                str(args.cache_items),
                "--metric",
                args.metric,
                "--eval_split",
                args.eval_split,
                "--n_trials",
                str(worker_trials),
                "--study_name",
                args.study_name,
                "--storage",
                args.storage,
                "--sampler_seed",
                str(args.sampler_seed + worker_index),
                "--skip_cache",
                "--hidden",
                str(args.hidden),
                "--num_layers",
                str(args.num_layers),
                "--kernel_size",
                str(args.kernel_size),
                "--dropout",
                str(args.dropout),
            ]
            if args.manifests_dir:
                cmd.extend(["--manifests_dir", args.manifests_dir])
            if args.split_groups:
                cmd.extend(["--split_groups", *args.split_groups])
            if args.thresholds:
                cmd.extend(["--thresholds", *args.thresholds])
            if args.batch_size is not None:
                cmd.extend(["--batch_size", str(args.batch_size)])
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
            if args.overwrite_runs:
                cmd.append("--overwrite_runs")
            if args.legacy_feature_dir:
                cmd.extend(["--legacy_feature_dir", args.legacy_feature_dir])
            if args.experimental_pdb_dir:
                cmd.extend(["--experimental_pdb_dir", args.experimental_pdb_dir])
            if args.alphafold_pdb_dir:
                cmd.extend(["--alphafold_pdb_dir", args.alphafold_pdb_dir])
            if args.esm3_pdb_dir:
                cmd.extend(["--esm3_pdb_dir", args.esm3_pdb_dir])
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env)
            procs.append(proc)
            if worker_index < len(args.gpus) - 1:
                time.sleep(2.0)

        failed = False
        for proc in procs:
            if proc.wait() != 0:
                failed = True
        if failed:
            raise RuntimeError("One or more Optuna workers failed")
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    main()
