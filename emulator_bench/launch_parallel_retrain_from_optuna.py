import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import DEFAULT_SPLIT_GROUPS, apply_manifest_paths_to_jobs, discover_split_jobs, normalize_threshold_args
from emulator_bench.run_split_benchmarks import maybe_cache_embeddings


TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def _load_config_payload(config_path: str | None) -> dict:
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Config JSON must contain a top-level object.")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrain DEKP across splits/seeds in parallel from Optuna best params.")
    parser.add_argument("--config_json", type=str, default=None)
    parser.add_argument("--gpus", nargs="+", default=None)
    parser.add_argument("--jobs_per_gpu", type=int, default=1)
    parser.add_argument(
        "--trials_per_gpu",
        dest="jobs_per_gpu",
        type=int,
        help="Alias for --jobs_per_gpu kept for compatibility with older commands.",
    )
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--manifests_dir", type=str, default=None)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--feature_list", type=str, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--preload_proteins", action="store_true")
    parser.add_argument("--preload_ligands", action="store_true")
    parser.add_argument("--preload_structures", action="store_true")
    parser.add_argument("--cache_items", type=int, default=64)
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=9)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
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
    return parser


def _load_best_hparams(args):
    payload = {}
    if args.storage:
        import optuna

        study = optuna.load_study(study_name=args.study_name, storage=args.storage)
        payload.update(study.best_params)
        payload["best_trial_number"] = int(study.best_trial.number)
        payload["best_value"] = float(study.best_value)
    if args.hparams_json:
        with open(args.hparams_json, "r", encoding="utf-8") as handle:
            json_payload = json.load(handle)
        if not isinstance(json_payload, dict):
            raise ValueError("Hyperparameter JSON must contain a top-level object.")
        payload.update(json_payload)
    payload.setdefault("batch_size", int(args.batch_size))
    payload.setdefault("lr", float(args.lr))
    payload.setdefault("weight_decay", float(args.weight_decay))
    payload.setdefault("dropout", float(args.dropout))
    return payload


def main():
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config_json", type=str, default=None)
    bootstrap_args, _ = bootstrap_parser.parse_known_args()

    parser = _build_parser()
    config_payload = _load_config_payload(bootstrap_args.config_json)
    parser.set_defaults(**config_payload)
    args = parser.parse_args()

    missing = [name for name in ["gpus", "base_dir", "cache_dir", "feature_list"] if getattr(args, name) in (None, [], "")]
    if missing:
        parser.error(f"Missing required arguments (via CLI or --config_json): {', '.join(missing)}")
    if int(args.jobs_per_gpu) < 1:
        parser.error("--jobs_per_gpu must be at least 1.")

    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    maybe_cache_embeddings(args)
    best_hparams = _load_best_hparams(args)
    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {args.base_dir}")
    if args.manifests_dir:
        jobs = apply_manifest_paths_to_jobs(jobs, Path(args.manifests_dir).expanduser(), require=True)

    work_queue = queue.Queue()
    for job in jobs:
        for seed in args.seeds:
            work_queue.put((job, seed))

    failures = []

    def worker(gpu_id: str, slot_idx: int):
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
                str(int(best_hparams["batch_size"])),
                "--epochs",
                str(args.epochs),
                "--lr",
                str(float(best_hparams["lr"])),
                "--weight_decay",
                str(float(best_hparams["weight_decay"])),
                "--hidden",
                str(args.hidden),
                "--num_layers",
                str(args.num_layers),
                "--kernel_size",
                str(args.kernel_size),
                "--dropout",
                str(float(best_hparams["dropout"])),
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
                failures.append((gpu_id, slot_idx, job["split_group"], job["split_name"], seed, return_code))
            work_queue.task_done()

    threads = []
    for gpu_id in args.gpus:
        for slot_idx in range(int(args.jobs_per_gpu)):
            threads.append(threading.Thread(target=worker, args=(gpu_id, slot_idx), daemon=True))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if failures:
        raise RuntimeError(f"Parallel retrain failures: {failures}")


if __name__ == "__main__":
    main()
