import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import DEFAULT_BASE_DIR
from emulator_bench.common import regression_metrics


DEFAULT_OUTPUT_DIR_NAME = "dekp_results_optuna_aggregated"
RESULT_DIR_NAME = "dekp_results_optuna"
SEED_RE = re.compile(r"^seed_(?P<seed>-?\d+)$")
METRIC_FILES = {
    "val": "final_results_val.csv",
    "test": "final_results_test.csv",
}
PREDICTION_FILES = {
    "val": "pred_label_val.csv",
    "test": "pred_label_test.csv",
}
COMPUTED_METRIC_COLUMNS = ["rmse", "mse", "mae", "r2", "pearson", "spearman"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate completed DEKP Optuna retrain outputs across every "
            "dekp_results_optuna folder under a split base directory."
        )
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help=f"Root containing split folders. Default: {DEFAULT_BASE_DIR}",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Directory for aggregate CSV files. Default: "
            f"<base_dir>/{DEFAULT_OUTPUT_DIR_NAME}"
        ),
    )
    parser.add_argument(
        "--result_dir_name",
        default=RESULT_DIR_NAME,
        help=f"Result directory name to discover. Default: {RESULT_DIR_NAME}",
    )
    parser.add_argument(
        "--skip_predictions",
        action="store_true",
        help="Skip aggregating pred_label_val.csv and pred_label_test.csv files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error if any discovered seed directory is missing expected files.",
    )
    return parser.parse_args()


def split_metadata(base_dir: Path, result_dir: Path) -> Dict[str, object]:
    split_dir = result_dir.parent
    rel_parts = split_dir.relative_to(base_dir).parts
    split_group = rel_parts[0] if rel_parts else split_dir.name
    threshold = None
    for part in rel_parts:
        if part.startswith("threshold_"):
            threshold = part.removeprefix("threshold_")
            break
    return {
        "split_path": split_dir.relative_to(base_dir).as_posix(),
        "split_group": split_group,
        "threshold": threshold,
        "result_dir": str(result_dir),
    }


def seed_from_dir(seed_dir: Path) -> Optional[int]:
    match = SEED_RE.match(seed_dir.name)
    if not match:
        return None
    return int(match.group("seed"))


def discover_result_dirs(base_dir: Path, result_dir_name: str) -> List[Path]:
    return sorted(path for path in base_dir.rglob(result_dir_name) if path.is_dir())


def discover_seed_dirs(result_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in result_dir.iterdir()
        if path.is_dir() and SEED_RE.match(path.name)
    )


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def resolve_label_column(df: pd.DataFrame) -> str:
    for column in ["label", "target"]:
        if column in df.columns:
            return column
    raise ValueError("Prediction file must contain a label or target column")


def compute_metrics_from_predictions(path: Path) -> tuple[Dict[str, float], Dict[str, object]]:
    df = read_csv(path)
    if df.empty:
        metrics = {column: float("nan") for column in COMPUTED_METRIC_COLUMNS}
        return metrics, {"valid_rows": 0, "reason": "empty prediction file"}
    if "prediction" not in df.columns:
        raise ValueError(f"Prediction file is missing a prediction column: {path}")
    label_col = resolve_label_column(df)
    values = df[[label_col, "prediction"]].apply(pd.to_numeric, errors="coerce").dropna()
    metrics = regression_metrics(
        values[label_col].to_numpy(),
        values["prediction"].to_numpy(),
    )
    label_std = values[label_col].std(ddof=0)
    prediction_std = values["prediction"].std(ddof=0)
    reason = ""
    if len(values) < 2:
        reason = "fewer than 2 valid prediction rows"
    elif label_std == 0:
        reason = "constant labels"
    elif prediction_std == 0:
        reason = "constant predictions"
    return metrics, {
        "valid_rows": len(values),
        "label_column": label_col,
        "label_nunique": values[label_col].nunique(dropna=True),
        "prediction_nunique": values["prediction"].nunique(dropna=True),
        "label_std": label_std,
        "prediction_std": prediction_std,
        "reason": reason,
    }


def is_missing_value(value: object) -> bool:
    return pd.isna(value)


def backfill_missing_metrics(
    df: pd.DataFrame,
    prediction_path: Path,
    metric_prefix: str = "",
) -> tuple[pd.DataFrame, List[Dict[str, object]]]:
    if df.empty or not prediction_path.exists():
        return df, []

    metrics, diagnostics = compute_metrics_from_predictions(prediction_path)
    out = df.copy()
    report_rows: List[Dict[str, object]] = []
    for metric_name, computed_value in metrics.items():
        column = f"{metric_prefix}{metric_name}"
        if column not in out.columns:
            out[column] = pd.NA

        missing_mask = out[column].map(is_missing_value)
        if not missing_mask.any():
            continue

        filled_count = 0
        if not pd.isna(computed_value):
            out.loc[missing_mask, column] = computed_value
            filled_count = int(missing_mask.sum())

        report_rows.append(
            {
                "metric_column": column,
                "computed_metric": metric_name,
                "computed_value": computed_value,
                "missing_rows": int(missing_mask.sum()),
                "filled_rows": filled_count,
                "remaining_missing_rows": int(missing_mask.sum()) - filled_count,
                "prediction_file": str(prediction_path),
                **diagnostics,
            }
        )

    return out, report_rows


def with_context(
    df: pd.DataFrame,
    metadata: Dict[str, object],
    seed_dir: Path,
    source_file: Path,
    extra: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    context = {
        **metadata,
        "seed_dir": str(seed_dir),
        "seed_from_dir": seed_from_dir(seed_dir),
        "source_file": str(source_file),
    }
    if extra:
        context.update(extra)

    out = df.copy()
    for key, value in reversed(list(context.items())):
        out.insert(0, key, value)
    return out


def record_missing(
    missing_rows: List[Dict[str, object]],
    metadata: Dict[str, object],
    seed_dir: Path,
    filename: str,
) -> None:
    missing_rows.append(
        {
            **metadata,
            "seed_dir": str(seed_dir),
            "seed_from_dir": seed_from_dir(seed_dir),
            "missing_file": filename,
        }
    )


def collect_frames(
    base_dir: Path,
    result_dir_name: str,
    include_predictions: bool,
) -> Dict[str, pd.DataFrame]:
    result_dirs = discover_result_dirs(base_dir, result_dir_name)
    if not result_dirs:
        raise FileNotFoundError(f"No {result_dir_name!r} directories found under {base_dir}")

    summary_frames: List[pd.DataFrame] = []
    logfile_frames: List[pd.DataFrame] = []
    metric_frames: List[pd.DataFrame] = []
    prediction_frames: List[pd.DataFrame] = []
    backfill_rows: List[Dict[str, object]] = []
    missing_rows: List[Dict[str, object]] = []
    discovered_rows: List[Dict[str, object]] = []

    for result_dir in result_dirs:
        metadata = split_metadata(base_dir, result_dir)
        seed_dirs = discover_seed_dirs(result_dir)
        discovered_rows.append(
            {
                **metadata,
                "seed_dir_count": len(seed_dirs),
            }
        )
        if not seed_dirs:
            record_missing(missing_rows, metadata, result_dir, "seed_* directory")
            continue

        for seed_dir in seed_dirs:
            summary_path = seed_dir / "run_summary.csv"
            if summary_path.exists():
                summary_df = read_csv(summary_path)
                for eval_split, prediction_filename in PREDICTION_FILES.items():
                    prediction_path = seed_dir / prediction_filename
                    summary_df, reports = backfill_missing_metrics(
                        summary_df,
                        prediction_path,
                        metric_prefix=f"{eval_split}_",
                    )
                    backfill_rows.extend(
                        with_report_context(
                            reports,
                            metadata,
                            seed_dir,
                            summary_path,
                            eval_split,
                        )
                    )
                summary_frames.append(with_context(summary_df, metadata, seed_dir, summary_path))
            else:
                record_missing(missing_rows, metadata, seed_dir, summary_path.name)

            logfile_path = seed_dir / "logfile.csv"
            if logfile_path.exists():
                logfile_frames.append(
                    with_context(read_csv(logfile_path), metadata, seed_dir, logfile_path)
                )
            else:
                record_missing(missing_rows, metadata, seed_dir, logfile_path.name)

            for eval_split, filename in METRIC_FILES.items():
                metric_path = seed_dir / filename
                if metric_path.exists():
                    metric_df = read_csv(metric_path)
                    prediction_path = seed_dir / PREDICTION_FILES[eval_split]
                    metric_df, reports = backfill_missing_metrics(metric_df, prediction_path)
                    backfill_rows.extend(
                        with_report_context(
                            reports,
                            metadata,
                            seed_dir,
                            metric_path,
                            eval_split,
                        )
                    )
                    metric_frames.append(
                        with_context(
                            metric_df,
                            metadata,
                            seed_dir,
                            metric_path,
                            {"eval_split": eval_split},
                        )
                    )
                else:
                    record_missing(missing_rows, metadata, seed_dir, filename)

            if include_predictions:
                for eval_split, filename in PREDICTION_FILES.items():
                    prediction_path = seed_dir / filename
                    if prediction_path.exists():
                        prediction_frames.append(
                            with_context(
                                read_csv(prediction_path),
                                metadata,
                                seed_dir,
                                prediction_path,
                                {"eval_split": eval_split},
                            )
                        )
                    else:
                        record_missing(missing_rows, metadata, seed_dir, filename)

    return {
        "discovered_result_dirs": pd.DataFrame(discovered_rows),
        "run_summaries": concat_or_empty(summary_frames),
        "logfiles": concat_or_empty(logfile_frames),
        "final_results": concat_or_empty(metric_frames),
        "predictions": concat_or_empty(prediction_frames),
        "metric_backfill_report": pd.DataFrame(backfill_rows),
        "missing_files": pd.DataFrame(missing_rows),
    }


def concat_or_empty(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def with_report_context(
    rows: List[Dict[str, object]],
    metadata: Dict[str, object],
    seed_dir: Path,
    source_file: Path,
    eval_split: str,
) -> List[Dict[str, object]]:
    return [
        {
            **metadata,
            "seed_dir": str(seed_dir),
            "seed_from_dir": seed_from_dir(seed_dir),
            "source_file": str(source_file),
            "eval_split": eval_split,
            **row,
        }
        for row in rows
    ]


def flatten_columns(columns: Iterable[object]) -> List[str]:
    flat = []
    for column in columns:
        if isinstance(column, tuple):
            flat.append("_".join(str(part) for part in column if part))
        else:
            flat.append(str(column))
    return flat


def summarize_metrics(final_results: pd.DataFrame) -> pd.DataFrame:
    if final_results.empty:
        return pd.DataFrame()

    group_cols = ["split_path", "split_group", "threshold", "eval_split"]
    skip_cols = set(group_cols) | {
        "result_dir",
        "seed_dir",
        "seed_from_dir",
        "source_file",
    }
    numeric_cols = [
        col
        for col in final_results.columns
        if col not in skip_cols and pd.api.types.is_numeric_dtype(final_results[col])
    ]
    if not numeric_cols:
        return pd.DataFrame()

    grouped = final_results.groupby(group_cols, dropna=False)[numeric_cols].agg(
        ["count", "mean", "std", "min", "max"]
    )
    grouped.columns = flatten_columns(grouped.columns)
    out = grouped.reset_index()
    return out.replace({math.nan: ""})


def write_outputs(frames: Dict[str, pd.DataFrame], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "discovered_result_dirs": output_dir / "discovered_result_dirs.csv",
        "run_summaries": output_dir / "aggregated_run_summaries.csv",
        "logfiles": output_dir / "aggregated_logfiles.csv",
        "final_results": output_dir / "aggregated_final_results.csv",
        "predictions": output_dir / "aggregated_predictions.csv",
        "metric_backfill_report": output_dir / "metric_backfill_report.csv",
        "missing_files": output_dir / "missing_files.csv",
        "split_metric_summary": output_dir / "split_metric_summary.csv",
    }

    frames = dict(frames)
    frames["split_metric_summary"] = summarize_metrics(frames["final_results"])

    written = []
    for name, path in paths.items():
        df = frames.get(name, pd.DataFrame())
        df.to_csv(path, index=False)
        written.append(path)
    return written


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else base_dir / DEFAULT_OUTPUT_DIR_NAME
    )

    frames = collect_frames(
        base_dir=base_dir,
        result_dir_name=args.result_dir_name,
        include_predictions=not args.skip_predictions,
    )
    written = write_outputs(frames, output_dir)

    missing_count = len(frames["missing_files"])
    if args.strict and missing_count:
        missing_path = output_dir / "missing_files.csv"
        raise SystemExit(
            f"Found {missing_count} missing expected files. See {missing_path}"
        )

    print(f"Wrote {len(written)} aggregate files to {output_dir}")
    print(f"Discovered result dirs: {len(frames['discovered_result_dirs'])}")
    print(f"Aggregated run summaries: {len(frames['run_summaries'])}")
    print(f"Aggregated logfile rows: {len(frames['logfiles'])}")
    print(f"Aggregated final result rows: {len(frames['final_results'])}")
    if not args.skip_predictions:
        print(f"Aggregated prediction rows: {len(frames['predictions'])}")
    print(f"Metric backfill rows: {len(frames['metric_backfill_report'])}")
    print(f"Missing expected files: {missing_count}")


if __name__ == "__main__":
    main()
