import argparse
import gc
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
try:
    from src.utils.rich_progress import progress, write
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.utils.rich_progress import progress, write

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    COMMON_PROTEIN_ID_COLS,
    COMMON_SEQUENCE_COLS,
    COMMON_SMILES_COLS,
    COMMON_TARGET_COLS,
    COMMON_STRUCTURE_ID_COLS,
    DEFAULT_BASE_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_FEATURES,
    append_csv_row,
    find_first_existing_column,
    ligand_cache_path,
    load_json,
    protein_cache_path,
    protein_sequence_cache_max_len,
    read_table,
    regression_metrics,
    resolve_single_split_job,
    save_json,
    set_seed,
    structure_cache_path,
    write_csv,
)
from emulator_bench.dataset import CachedDEKPDataset, LigandEmbeddingStore, ProteinEmbeddingStore, StructureEmbeddingStore
from emulator_bench.modeling import MetaDecoder, graph_collate_fn


def _resolve_amp(device: torch.device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, "fp32"
    index = device.index if device.index is not None else torch.cuda.current_device()
    major, _ = torch.cuda.get_device_capability(index)
    if major >= 8:
        return torch.bfloat16, "bf16"
    return torch.float16, "fp16"


def _autocast_context(device: torch.device, amp_dtype):
    if device.type == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def _worker_init_fn(worker_id: int) -> None:
    # Each structure cache file is ~4 MB. With the default LRU (cache_items) and
    # prefetch_factor the DataLoader can hold many GB of graph tensors simultaneously.
    # Run Python's cyclic GC more aggressively in workers so that any Python-level
    # cycles (e.g. from future torch_geometric upgrades) don't accumulate.
    import gc as _gc
    _gc.set_threshold(100, 5, 2)


def _resolve_columns(frame: pd.DataFrame, manifest: dict, args):
    resolved = manifest.get("resolved_columns", {})
    pdb_record_col = resolved.get("pdb_record_col") or ("pdbs" if "pdbs" in frame.columns else None)
    manifest_structure_id = resolved.get("structure_id_col")
    if manifest_structure_id == resolved.get("sequence_col"):
        manifest_structure_id = None  # was a sequence fallback in the manifest, not a real structure column
    structure_id_col = find_first_existing_column(frame, COMMON_STRUCTURE_ID_COLS, explicit=args.structure_id_col or manifest_structure_id, required=False)
    if structure_id_col is None and pdb_record_col:
        structure_id_col = pdb_record_col
    return {
        "sequence_col": find_first_existing_column(frame, COMMON_SEQUENCE_COLS, explicit=args.sequence_col or resolved.get("sequence_col"), required=True),
        "smiles_col": find_first_existing_column(frame, COMMON_SMILES_COLS, explicit=args.smiles_col or resolved.get("smiles_col"), required=True),
        "protein_id_col": find_first_existing_column(frame, COMMON_PROTEIN_ID_COLS, explicit=args.protein_id_col or resolved.get("protein_id_col"), required=False),
        "structure_id_col": structure_id_col,
        "target_col": find_first_existing_column(frame, COMMON_TARGET_COLS, explicit=args.target_col or resolved.get("target_col"), required=True),
        "pdb_record_col": pdb_record_col,
    }


def _make_loader(dataset, batch_size, shuffle, args):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "collate_fn": graph_collate_fn,
        "drop_last": False,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
        kwargs["worker_init_fn"] = _worker_init_fn
    return DataLoader(**kwargs)


def _move_batch(graph_batch, protein_tokens, smiles_tokens, features, labels, device):
    graph_batch = graph_batch.to(device)
    protein_tokens = protein_tokens.to(device, non_blocking=True)
    smiles_tokens = smiles_tokens.to(device, non_blocking=True)
    features = features.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    return graph_batch, protein_tokens, smiles_tokens, features, labels


def evaluate(model, loader, device, amp_dtype):
    model.eval()
    mse_loss = torch.nn.MSELoss(reduction="mean")
    total_loss = 0.0
    total_examples = 0
    preds, labels, metadata_rows = [], [], []

    with torch.no_grad():
        for graph_batch, protein_tokens, smiles_tokens, features, label_tensor, metadata in loader:
            graph_batch, protein_tokens, smiles_tokens, features, label_tensor = _move_batch(
                graph_batch, protein_tokens, smiles_tokens, features, label_tensor, device
            )
            with _autocast_context(device, amp_dtype):
                outputs = model(graph_batch, protein_tokens, smiles_tokens, features)
                loss = mse_loss(outputs, label_tensor)

            batch_size = int(label_tensor.numel())
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            preds.extend(outputs.detach().float().cpu().tolist())
            labels.extend(label_tensor.detach().float().cpu().tolist())
            metadata_rows.extend(metadata)

    metrics = regression_metrics(labels, preds)
    metrics["loss"] = total_loss / max(1, total_examples)
    return metrics, preds, labels, metadata_rows


def _save_checkpoint(path: Path, model, optimizer, epoch: int, args, manifest: dict, feature_dim_list, best_metric: float):
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "cache_manifest": manifest,
        "feature_dim_list": feature_dim_list,
        "best_metric": float(best_metric),
    }
    torch.save(payload, path)


def _write_prediction_csv(path: Path, preds, labels, metadata_rows):
    rows = []
    for pred, label, metadata in zip(preds, labels, metadata_rows):
        row = dict(metadata)
        row["label"] = label
        row["prediction"] = pred
        rows.append(row)
    write_csv(path, rows)


def _resolve_feature_dims(manifest: dict, feature_names, train_ds) -> list[int]:
    feature_dims = dict(manifest.get("feature_dims", {}))
    missing = [name for name in feature_names if name not in feature_dims]
    if missing:
        sample_row = train_ds.frame.iloc[0]
        sequence = str(sample_row[train_ds.sequence_col])
        smiles = str(sample_row[train_ds.smiles_col])
        structure_id = str(sample_row[train_ds.structure_id_col]) if train_ds.structure_id_col in sample_row.index else str(sample_row[train_ds.protein_id_col])
        protein_payload = train_ds.protein_store.get(sequence)
        ligand_payload = train_ds.ligand_store.get(smiles)
        structure_payload = train_ds.structure_store.get(structure_id, sequence)
        for name in missing:
            if name in protein_payload:
                feature_dims[name] = int(torch.as_tensor(protein_payload[name]).numel())
            elif name in ligand_payload:
                feature_dims[name] = int(torch.as_tensor(ligand_payload[name]).numel())
            elif name in structure_payload:
                feature_dims[name] = int(torch.as_tensor(structure_payload[name]).numel())
            else:
                raise KeyError(f"Requested feature `{name}` is missing from both the cache manifest and cached payloads.")
    return [int(feature_dims[name]) for name in feature_names]


def _filter_missing_cache_rows(frame: pd.DataFrame, split_name: str, resolved_columns: dict, cache_dir: Path, manifest: dict) -> pd.DataFrame:
    sequence_col = resolved_columns["sequence_col"]
    protein_id_col = resolved_columns["protein_id_col"]
    structure_id_col = resolved_columns["structure_id_col"]
    protein_max_len = int(manifest["protein_max_len"])
    protein_cache_max_len = protein_sequence_cache_max_len(protein_max_len)

    keep_mask = []
    missing_protein_ids = []
    missing_ligand_ids = []
    missing_structure_ids = []
    for _, row in frame.iterrows():
        sequence = str(row[sequence_col])
        smiles = str(row[resolved_columns["smiles_col"]])
        protein_id = str(row[protein_id_col])
        structure_value = row[structure_id_col] if structure_id_col in row.index else None
        if structure_value is None or pd.isna(structure_value) or str(structure_value).strip().lower() in ("", "nan", "none"):
            structure_id = protein_id
        else:
            structure_id = str(structure_value).strip()

        protein_path = protein_cache_path(cache_dir, sequence=sequence, max_len=protein_cache_max_len)
        ligand_path = ligand_cache_path(cache_dir, smiles=smiles)
        structure_path = structure_cache_path(cache_dir, structure_id=structure_id, fallback_sequence=sequence)

        protein_exists = protein_path.exists()
        ligand_exists = ligand_path.exists()
        structure_exists = structure_path.exists()
        keep_mask.append(protein_exists and ligand_exists and structure_exists)
        if not protein_exists:
            missing_protein_ids.append(protein_id)
        if not ligand_exists:
            missing_ligand_ids.append(smiles)
        if not structure_exists:
            missing_structure_ids.append(structure_id)

    total_rows = int(len(frame))
    removed_rows = int(total_rows - sum(keep_mask))
    kept_rows = int(total_rows - removed_rows)
    removed_pct = (100.0 * removed_rows / total_rows) if total_rows else 0.0
    payload = {
        "split": split_name,
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "removed_rows_missing_any_cache": removed_rows,
        "removed_pct_missing_any_cache": round(removed_pct, 4),
        "rows_missing_protein_cache": len(missing_protein_ids),
        "rows_missing_ligand_cache": len(missing_ligand_ids),
        "rows_missing_structure_cache": len(missing_structure_ids),
    }
    if missing_protein_ids:
        payload["missing_protein_examples"] = missing_protein_ids[:10]
    if missing_ligand_ids:
        payload["missing_ligand_examples"] = missing_ligand_ids[:10]
    if missing_structure_ids:
        payload["missing_structure_examples"] = missing_structure_ids[:10]
    print(json.dumps(payload), flush=True)

    return frame.loc[keep_mask].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Train DEKP on explicit train/val/test splits with cached features.")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--split_group", type=str, default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="dekp_run")
    parser.add_argument("--feature_list", type=str, default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--sequence_col", type=str, default=None)
    parser.add_argument("--smiles_col", type=str, default=None)
    parser.add_argument("--protein_id_col", type=str, default=None)
    parser.add_argument("--structure_id_col", type=str, default=None)
    parser.add_argument("--target_col", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=9)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--preload_proteins", action="store_true")
    parser.add_argument("--preload_ligands", action="store_true")
    parser.add_argument("--preload_structures", action="store_true")
    parser.add_argument("--cache_items", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--compile_model", action="store_true")
    args = parser.parse_args()

    if args.train_path is None or args.val_path is None or args.test_path is None:
        if not args.split_group:
            raise ValueError("Provide either explicit --train_path/--val_path/--test_path or --split_group.")
        job = resolve_single_split_job(Path(args.base_dir), args.split_group, args.threshold)
        args.train_path = job["train_path"]
        args.val_path = job["val_path"]
        args.test_path = job["test_path"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    manifest = load_json(cache_dir / "manifest.json")
    args.feature_list = [item.strip() for item in args.feature_list.split(",") if item.strip()]

    train_df = read_table(Path(args.train_path))
    val_df = read_table(Path(args.val_path))
    test_df = read_table(Path(args.test_path))
    resolved_columns = _resolve_columns(train_df, manifest, args)
    if resolved_columns["protein_id_col"] is None:
        resolved_columns["protein_id_col"] = resolved_columns["sequence_col"]
    if resolved_columns["structure_id_col"] is None:
        resolved_columns["structure_id_col"] = resolved_columns["protein_id_col"]

    original_train_rows = len(train_df)
    original_val_rows = len(val_df)
    original_test_rows = len(test_df)

    train_df = _filter_missing_cache_rows(train_df, "train", resolved_columns, cache_dir, manifest)
    val_df = _filter_missing_cache_rows(val_df, "val", resolved_columns, cache_dir, manifest)
    test_df = _filter_missing_cache_rows(test_df, "test", resolved_columns, cache_dir, manifest)

    filtered_total = len(train_df) + len(val_df) + len(test_df)
    original_total = original_train_rows + original_val_rows + original_test_rows
    removed_total = int(original_total - filtered_total)
    removed_total_pct = (100.0 * removed_total / original_total) if original_total else 0.0
    print(
        json.dumps(
            {
                "split_filter_summary": {
                    "total_rows": int(original_total),
                    "kept_rows": int(filtered_total),
                    "removed_rows_missing_any_cache": removed_total,
                    "removed_pct_missing_any_cache": round(removed_total_pct, 4),
                }
            }
        ),
        flush=True,
    )
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError(
            f"After dropping rows with missing cache entries, split sizes are train={len(train_df)}, val={len(val_df)}, test={len(test_df)}."
        )

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    amp_dtype, amp_name = _resolve_amp(device)

    protein_store = ProteinEmbeddingStore(
        cache_dir=cache_dir,
        sequences=train_df[resolved_columns["sequence_col"]].tolist() if args.preload_proteins else None,
        preload=args.preload_proteins,
        max_items=args.cache_items,
        max_len=int(manifest["protein_max_len"]),
    )
    ligand_store = LigandEmbeddingStore(
        cache_dir=cache_dir,
        smiles_values=train_df[resolved_columns["smiles_col"]].tolist() if args.preload_ligands else None,
        preload=args.preload_ligands,
        max_items=args.cache_items,
    )
    structure_items = None
    if args.preload_structures:
        structure_items = list(
            zip(
                train_df[resolved_columns["structure_id_col"]].astype(str).tolist(),
                train_df[resolved_columns["sequence_col"]].astype(str).tolist(),
            )
        )
    structure_store = StructureEmbeddingStore(
        cache_dir=cache_dir,
        items=structure_items,
        preload=args.preload_structures,
        max_items=args.cache_items,
    )

    dataset_kwargs = {
        "protein_store": protein_store,
        "ligand_store": ligand_store,
        "structure_store": structure_store,
        "feature_names": args.feature_list,
        "sequence_col": resolved_columns["sequence_col"],
        "smiles_col": resolved_columns["smiles_col"],
        "protein_id_col": resolved_columns["protein_id_col"],
        "structure_id_col": resolved_columns["structure_id_col"],
        "target_col": resolved_columns["target_col"],
        "protein_max_len": int(manifest["protein_max_len"]),
        "smiles_max_len": int(manifest["smiles_max_len"]),
        "protein_pad_id": int(manifest["protein_pad_id"]),
        "smiles_pad_id": int(manifest["smiles_pad_id"]),
    }
    train_ds = CachedDEKPDataset(train_df, **dataset_kwargs)
    val_ds = CachedDEKPDataset(val_df, **dataset_kwargs)
    test_ds = CachedDEKPDataset(test_df, **dataset_kwargs)

    train_loader = _make_loader(train_ds, batch_size=args.batch_size, shuffle=True, args=args)
    val_loader = _make_loader(val_ds, batch_size=args.batch_size, shuffle=False, args=args)
    test_loader = _make_loader(test_ds, batch_size=args.batch_size, shuffle=False, args=args)

    feature_dim_list = _resolve_feature_dims(manifest, args.feature_list, train_ds)
    model = MetaDecoder(
        seq_vocab_size=int(manifest["protein_vocab_size"]),
        smi_vocab_size=int(manifest["smiles_vocab_size"]),
        feature_dim_list=feature_dim_list,
        hidden=args.hidden,
        num_layers=args.num_layers,
        protein_len=int(manifest["protein_max_len"]),
        smi_len=int(manifest["smiles_max_len"]),
        dropout=args.dropout,
        kernel_size=args.kernel_size,
    ).to(device)
    if args.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler_enabled = device.type == "cuda" and amp_dtype == torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    started = time.time()
    log_path = out_dir / "logfile.csv"

    with progress(total=args.epochs, desc="Training", unit="epoch", leave=True) as epoch_bar:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_examples = 0
            batch_count = 0
            iterator = progress(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch", leave=False)
            for graph_batch, protein_tokens, smiles_tokens, features, labels, _ in iterator:
                graph_batch, protein_tokens, smiles_tokens, features, labels = _move_batch(
                    graph_batch, protein_tokens, smiles_tokens, features, labels, device
                )
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(device, amp_dtype):
                    outputs = model(graph_batch, protein_tokens, smiles_tokens, features)
                    loss = criterion(outputs, labels)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                batch_size = int(labels.numel())
                train_loss_sum += float(loss.item()) * batch_size
                train_examples += batch_size
                batch_count += 1
                if batch_count % 100 == 0:
                    gc.collect()
                iterator.set_postfix(loss=f"{(train_loss_sum / max(1, train_examples)):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            del iterator

            train_loss = train_loss_sum / max(1, train_examples)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
            append_csv_row(log_path, row)

            if epoch % 10 == 0 or epoch == args.epochs:
                _save_checkpoint(out_dir / "checkpoint_last.pt", model, optimizer, epoch, args, manifest, feature_dim_list, float("nan"))
            epoch_bar.set_postfix(loss=f"{train_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            epoch_bar.update(1)
            gc.collect()

    val_metrics, val_preds, val_labels, val_metadata = evaluate(model, val_loader, device=device, amp_dtype=amp_dtype)
    test_metrics, test_preds, test_labels, test_metadata = evaluate(model, test_loader, device=device, amp_dtype=amp_dtype)
    final_summary = {
        "task_name": args.task_name,
        "seed": args.seed,
        "amp_mode": amp_name,
        "elapsed_seconds": time.time() - started,
        "feature_list": ",".join(args.feature_list),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }

    write_csv(out_dir / "final_results_val.csv", [val_metrics])
    write_csv(out_dir / "final_results_test.csv", [test_metrics])
    _write_prediction_csv(out_dir / "pred_label_val.csv", val_preds, val_labels, val_metadata)
    _write_prediction_csv(out_dir / "pred_label_test.csv", test_preds, test_labels, test_metadata)
    write_csv(out_dir / "run_summary.csv", [final_summary])
    save_json(out_dir / "run_summary.json", final_summary)


if __name__ == "__main__":
    main()
