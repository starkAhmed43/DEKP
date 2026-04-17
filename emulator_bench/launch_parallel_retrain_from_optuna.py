import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import optuna

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import DEFAULT_SPLIT_GROUPS, discover_split_jobs, normalize_threshold_args
from emulator_bench.run_split_benchmarks import maybe_cache_embeddings


TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def _load_best_hparams(args):
    if args.hparams_json:
        with open(args.hparams_json, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if not args.storage:
        raise ValueError("Provide either --hparams_json or --storage.")
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    payload = dict(study.best_params)
    payload["best_trial_number"] = int(study.best_trial.number)
    payload["best_value"] = float(study.best_value)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Retrain DEKP across splits/seeds in parallel from Optuna best params.")
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--feature_list", type=str, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407])
    parser.add_argument("--epochs", type=int, default=80)
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
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=9)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--hparams_json", type=str, default=None)
    parser.add_argument("--study_name", type=str, default="dekp_optuna")
    parser.add_argument("--storage", type=str, default=None)
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
    parser.add_argument("--sequence_col", type=str, default=None)
    parser.add_argument("--smiles_col", type=str, default=None)
    parser.add_argument("--protein_id_col", type=str, default=None)
    parser.add_argument("--structure_id_col", type=str, default=None)
    parser.add_argument("--target_col", type=str, default=None)
    args = parser.parse_args()

    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    maybe_cache_embeddings(args)
    best_hparams = _load_best_hparams(args)
    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {args.base_dir}")

    work_queue = queue.Queue()
    for job in jobs:
        for seed in args.seeds:
            work_queue.put((job, seed))

    failures = []

    def worker(gpu_id: str):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        while True:
            try:
                job, seed = work_queue.get_nowait()
            except queue.Empty:
                return
            out_dir = Path(job["root_dir"]) / "dekp_results_optuna" / f"seed_{seed}"
            if out_dir.exists() and not args.overwrite_runs and (out_dir / "final_results_test.csv").exists():
                work_queue.task_done()
                continue
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
                str(out_dir),
                "--task_name",
                f"{job['split_group']}_{job['split_name']}_seed{seed}",
                "--feature_list",
                args.feature_list,
                "--batch_size",
                str(int(best_hparams.get("batch_size", 128))),
                "--epochs",
                str(args.epochs),
                "--lr",
                str(float(best_hparams.get("lr", 1e-3))),
                "--weight_decay",
                str(float(best_hparams.get("weight_decay", 3e-4))),
                "--scheduler",
                str(best_hparams.get("scheduler", "cosine")),
                "--lr_decay_factor",
                str(float(best_hparams.get("lr_decay_factor", 0.5))),
                "--lr_decay_patience",
                str(int(best_hparams.get("lr_decay_patience", 5))),
                "--min_lr",
                str(float(best_hparams.get("min_lr", 1e-6))),
                "--lr_warmup_epochs",
                str(int(best_hparams.get("lr_warmup_epochs", 3))),
                "--lr_warmup_start_factor",
                str(float(best_hparams.get("lr_warmup_start_factor", 0.1))),
                "--clip_grad",
                str(float(best_hparams.get("clip_grad", 1.0))),
                "--patience",
                str(int(best_hparams.get("patience", args.patience))),
                "--min_delta",
                str(float(best_hparams.get("min_delta", args.min_delta))),
                "--hidden",
                str(args.hidden),
                "--num_layers",
                str(args.num_layers),
                "--kernel_size",
                str(args.kernel_size),
                "--dropout",
                str(float(best_hparams.get("dropout", args.dropout))),
                "--device",
                "cuda:0",
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
            return_code = subprocess.call(cmd, cwd=str(REPO_ROOT), env=env)
            if return_code != 0:
                failures.append((gpu_id, job["split_group"], job["split_name"], seed, return_code))
            work_queue.task_done()

    threads = [threading.Thread(target=worker, args=(gpu_id,), daemon=True) for gpu_id in args.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if failures:
        raise RuntimeError(f"Parallel retrain failures: {failures}")


if __name__ == "__main__":
    main()
