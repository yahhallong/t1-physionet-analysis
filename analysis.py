"""
Utilities for exploring and analyzing T1 imputation results for the PhysioNet 2012 dataset.
The module is intentionally flexible to accommodate different baseline models and file formats
that may live inside the provided ZIP archive.
"""
from __future__ import annotations

import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency in this environment
    plt = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency in this environment
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency in this environment
    pd = None

def _core_dependencies_available() -> bool:
    if np is None or pd is None:
        print("[WARN] numpy and pandas are required for this analysis.")
        return False
    return True

def _ensure_core_dependencies() -> None:
    if np is None or pd is None:
        raise ImportError(
            "numpy and pandas are required. Install them to enable data loading and analysis."
        )

if np is not None:
    ArrayLike = Union[np.ndarray, Mapping[str, np.ndarray]]
else:  # Fallback typing when numpy is unavailable during import time.
    ArrayLike = Mapping[str, Any]


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

def reconstruct_zip_from_parts(parts_pattern: str, output_path: Path) -> Path:
    """Reconstruct a split ZIP archive.

    Parameters
    ----------
    parts_pattern:
        Glob pattern to locate all split parts (e.g., ``"physio_part_*"``).
    output_path:
        Path where the reconstructed ZIP should be written.

    Returns
    -------
    Path
        The path to the reconstructed ZIP archive.
    """

    parts = sorted(Path().glob(parts_pattern))
    if not parts:
        raise FileNotFoundError(f"No parts matched pattern: {parts_pattern}")

    output_path = Path(output_path)
    if output_path.exists():
        return output_path

    with output_path.open("wb") as outfile:
        for part in parts:
            with part.open("rb") as infile:
                outfile.write(infile.read())
    return output_path


def is_git_lfs_pointer(file_path: Path) -> bool:
    """Check whether a file is a Git LFS pointer rather than the real payload."""

    try:
        with file_path.open("r", encoding="utf-8") as f:
            first_line = f.readline()
            return first_line.startswith("version https://git-lfs.github.com/spec/v1")
    except UnicodeDecodeError:
        return False
    except FileNotFoundError:
        return False


def list_zip_contents(zip_path: Path) -> List[str]:
    """List contents of a ZIP archive safely."""

    with zipfile.ZipFile(zip_path, "r") as zf:
        return zf.namelist()


def extract_zip(zip_path: Path, output_dir: Path) -> Path:
    """Extract a ZIP archive into ``output_dir``.

    Extraction is skipped if the directory already exists.
    """

    output_dir = Path(output_dir)
    if output_dir.exists():
        return output_dir

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    return output_dir


def describe_directory_tree(root: Path, max_depth: int = 3) -> str:
    """Return a tree-like representation of a directory.

    The function avoids expensive recursive listings by respecting ``max_depth``.
    """

    root = Path(root)
    lines: List[str] = [root.name + "/"]

    for current_root, dirs, files in os.walk(root):
        rel_root = Path(current_root).relative_to(root)
        depth = len(rel_root.parts)
        if depth >= max_depth:
            dirs[:] = []
        indent = "    " * depth
        for d in sorted(dirs):
            lines.append(f"{indent}{d}/")
        for f in sorted(files):
            lines.append(f"{indent}{f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_data_info(path: Path) -> Union[dict, pd.DataFrame]:
    """Load ``data_info.json`` and return either the raw dict or a DataFrame.

    The function checks for common keys (``variables``) to construct a table of
    variable metadata if present; otherwise it returns the raw dictionary.
    """

    _ensure_core_dependencies()
    info = load_json(path)
    if isinstance(info, dict) and "variables" in info:
        df = pd.DataFrame(info["variables"])
        if "variable_id" not in df.columns and "id" in df.columns:
            df = df.rename(columns={"id": "variable_id"})
        return df
    return info


def _load_np_structure(file_path: Path) -> ArrayLike:
    """Load ``.npz`` or ``.npy`` files, returning arrays or dictionaries.

    ``allow_pickle=True`` is used to maximize compatibility with legacy files.
    """

    _ensure_core_dependencies()
    suffix = file_path.suffix.lower()
    if suffix == ".npz":
        data = np.load(file_path, allow_pickle=True)
        if len(data.files) == 1:
            return data[data.files[0]]
        return {k: data[k] for k in data.files}
    if suffix == ".npy":
        arr = np.load(file_path, allow_pickle=True)
        if arr.dtype == object and arr.size == 1:
            obj = arr.item()
            if isinstance(obj, (dict, np.ndarray)):
                return obj
        return arr
    raise ValueError(f"Unsupported array format: {file_path}")


def _infer_data_and_mask(structure: ArrayLike) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Infer data and mask arrays from a loaded structure.

    Parameters
    ----------
    structure:
        Output from ``_load_np_structure`` or similar. It can be a plain array
        or a mapping with keys like ``data``/``mask``.

    Returns
    -------
    data: np.ndarray
    mask: np.ndarray
    metadata: dict
    """

    metadata: dict = {}
    if isinstance(structure, Mapping):
        lower_keys = {k.lower(): k for k in structure.keys()}
        data_key = lower_keys.get("data")
        mask_key = lower_keys.get("mask")
        data = structure[data_key] if data_key else np.asarray(next(iter(structure.values())))
        mask = structure.get(mask_key) if mask_key else None
        metadata = {k: v for k, v in structure.items() if k not in {data_key, mask_key}}
    else:
        data = np.asarray(structure)
        mask = None

    if mask is None:
        mask = ~np.isnan(data)
    return data, np.asarray(mask, dtype=bool), metadata


def load_ground_truth(gt_dir: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load ground-truth time series data and masks.

    The function searches for ``.npz`` or ``.npy`` files first. If no binary
    files are found, it attempts to read CSV files. When no explicit mask is
    available, missing values are inferred using ``NaN`` entries in the data.
    """

    _ensure_core_dependencies()
    gt_dir = Path(gt_dir)
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

    binary_files = sorted(gt_dir.glob("*.npz")) + sorted(gt_dir.glob("*.npy"))
    if binary_files:
        structure = _load_np_structure(binary_files[0])
        return _infer_data_and_mask(structure)

    csv_files = sorted(gt_dir.glob("*.csv"))
    if csv_files:
        df = pd.read_csv(csv_files[0])
        data = df.to_numpy()
        mask = ~np.isnan(data)
        metadata = {"columns": df.columns.tolist()}
        return data, mask, metadata

    raise FileNotFoundError(f"No recognizable ground truth files in {gt_dir}")


def load_metrics_summary(root: Path) -> pd.DataFrame:
    """Load metrics summary (prefers ``metrics_summary.csv``).

    If the preferred file is missing, falls back to ``metrics_aggregate.csv``.
    """

    _ensure_core_dependencies()
    root = Path(root)
    for name in ["metrics_summary.csv", "metrics_aggregate.csv", "metrics_per_sample.csv"]:
        candidate = root / name
        if candidate.exists():
            return pd.read_csv(candidate)
    raise FileNotFoundError("No metrics CSV found; expected metrics_summary.csv or similar.")


def load_t1_predictions(results_root: Path) -> Dict[float, ArrayLike]:
    """Load T1 predictions grouped by additional missing ratios.

    The function scans for directories matching ``results_mrXX`` where ``XX``
    denotes the percentage of additional missingness. It loads the first
    ``.npz``/``.npy``/``.csv`` file found within each directory.
    """

    _ensure_core_dependencies()
    results_root = Path(results_root)
    preds: Dict[float, ArrayLike] = {}
    pattern = re.compile(r"results_mr(\d+)")

    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if not match:
            continue
        pct = int(match.group(1))
        ratio = pct / 100.0

        binary_files = sorted(child.glob("*.npz")) + sorted(child.glob("*.npy"))
        csv_files = sorted(child.glob("*.csv"))
        if binary_files:
            preds[ratio] = _load_np_structure(binary_files[0])
            continue
        if csv_files:
            preds[ratio] = pd.read_csv(csv_files[0])
            continue
        # Keep an empty placeholder if no files are found so that callers can handle gracefully.
        preds[ratio] = None
    return preds


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def compute_variable_missing_rates(gt_data: np.ndarray, gt_mask: np.ndarray) -> pd.DataFrame:
    """Compute natural missing rates per variable."""

    _ensure_core_dependencies()
    if gt_data.ndim < 2:
        raise ValueError("Ground truth data must be at least 2D (time x variables).")

    # Assume data shape [..., time, variables] and flatten all leading axes except variables.
    flat_mask = gt_mask.reshape(-1, gt_mask.shape[-1])
    missing_rate = 1.0 - flat_mask.mean(axis=0)
    df = pd.DataFrame({
        "variable_id": np.arange(flat_mask.shape[1]),
        "missing_rate": missing_rate,
    })
    return df


def compute_mse_per_variable(gt_data: np.ndarray, pred_data: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    """Compute per-variable MSE using observed ground-truth positions."""

    _ensure_core_dependencies()
    if gt_data.shape != pred_data.shape:
        raise ValueError("Ground truth and prediction shapes must match to compute MSE.")
    if mask.shape != gt_data.shape:
        raise ValueError("Mask shape must match ground truth.")

    obs = mask
    diff = (pred_data - gt_data) ** 2
    diff_masked = np.where(obs, diff, np.nan)
    mse = np.nanmean(diff_masked.reshape(-1, diff_masked.shape[-1]), axis=0)
    return pd.DataFrame({"variable_id": np.arange(diff_masked.shape[-1]), "mse": mse})


def group_variables_by_missing_rate(missing_df: pd.DataFrame) -> Dict[str, List[int]]:
    """Group variables into missingness buckets."""

    groups = {"low": [], "medium": [], "high": [], "extreme": []}
    for _, row in missing_df.iterrows():
        rate = row["missing_rate"]
        vid = int(row["variable_id"])
        if rate < 0.2:
            groups["low"].append(vid)
        elif rate <= 0.6:
            groups["medium"].append(vid)
        elif rate <= 0.9:
            groups["high"].append(vid)
        else:
            groups["extreme"].append(vid)
    return groups


def compute_group_mse_by_ratio(
    gt_data: np.ndarray,
    t1_preds_by_ratio: Mapping[float, ArrayLike],
    gt_mask: np.ndarray,
    var_groups: Mapping[str, List[int]],
) -> Dict[str, Dict[float, float]]:
    """Compute average MSE per variable group across ratios."""

    _ensure_core_dependencies()
    group_mse: Dict[str, Dict[float, float]] = {k: {} for k in var_groups}
    for ratio, preds in t1_preds_by_ratio.items():
        if preds is None:
            continue
        pred_data, pred_mask, _ = _infer_data_and_mask(preds)
        aligned_mask = gt_mask & pred_mask
        per_var = compute_mse_per_variable(gt_data, pred_data, aligned_mask)
        for group_name, vids in var_groups.items():
            if not vids:
                group_mse[group_name][ratio] = np.nan
                continue
            mse_vals = per_var.loc[per_var["variable_id"].isin(vids), "mse"]
            group_mse[group_name][ratio] = mse_vals.mean()
    return group_mse


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def _matplotlib_available() -> bool:
    if plt is None:
        print("[WARN] Matplotlib is not installed; skipping plot.")
        return False
    return True


def _ensure_output_dir(path: Optional[Path]) -> None:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def _attempt_plot(description: str, plot_func, *args, savepath: Optional[Path] = None, **kwargs) -> None:
    """Helper to call a plotting function and report status to the user."""

    try:
        plot_func(*args, savepath=savepath, **kwargs)
        if plt is None:
            print(f"[WARN] Matplotlib missing; skipped {description}.")
        else:
            print(f"[INFO] Saved {description} to {savepath}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not generate {description}: {exc}")


def plot_overall_degradation(model_results: Dict[str, Dict[float, float]], savepath: Optional[Path] = None) -> None:
    """Plot overall MSE degradation across additional missing ratios."""

    if not _matplotlib_available():
        return
    plt.figure(figsize=(6, 4))
    for model_name, ratios in model_results.items():
        xs = sorted(ratios.keys())
        ys = [ratios[r] for r in xs]
        plt.plot(xs, ys, marker="o", label=model_name)
    plt.xlabel("Additional missing ratio")
    plt.ylabel("MSE")
    plt.title("Overall performance degradation")
    plt.legend()
    plt.tight_layout()
    if savepath:
        _ensure_output_dir(Path(savepath))
        plt.savefig(savepath)
    plt.close()


def plot_variable_group_degradation(group_mse: Mapping[str, Mapping[float, float]], savepath: Optional[Path] = None) -> None:
    """Plot degradation per variable missingness group."""

    if not _matplotlib_available():
        return
    plt.figure(figsize=(6, 4))
    for group, ratios in group_mse.items():
        xs = sorted(ratios.keys())
        ys = [ratios[r] for r in xs]
        plt.plot(xs, ys, marker="o", label=group)
    plt.xlabel("Additional missing ratio")
    plt.ylabel("MSE")
    plt.title("Variable-group robustness")
    plt.legend(title="Natural missing group")
    plt.tight_layout()
    if savepath:
        _ensure_output_dir(Path(savepath))
        plt.savefig(savepath)
    plt.close()


def plot_extreme_case_mse(extreme_var_ids: Iterable[int], mse_by_var_and_ratio: Mapping[float, Mapping[int, float]], savepath: Optional[Path] = None) -> None:
    """Plot MSE for variables with natural missing rate > 0.9."""

    if not _matplotlib_available():
        return
    plt.figure(figsize=(6, 4))
    for vid in extreme_var_ids:
        xs = sorted(mse_by_var_and_ratio.keys())
        ys = [mse_by_var_and_ratio[r].get(vid, np.nan) for r in xs]
        plt.plot(xs, ys, marker="o", label=f"Var {vid}")
    plt.xlabel("Additional missing ratio")
    plt.ylabel("MSE")
    plt.title("Extreme-missing variables")
    plt.legend(title="Variable ID")
    plt.tight_layout()
    if savepath:
        _ensure_output_dir(Path(savepath))
        plt.savefig(savepath)
    plt.close()


def plot_single_variable_timeseries(
    gt_series: np.ndarray,
    pred_series: np.ndarray,
    mask: np.ndarray,
    title: str,
    savepath: Optional[Path] = None,
) -> None:
    """Overlay ground truth and predictions for a single variable/time series."""

    if not _matplotlib_available():
        return
    plt.figure(figsize=(8, 3))
    timesteps = np.arange(len(gt_series))
    plt.plot(timesteps, gt_series, label="Ground truth", linewidth=2)
    plt.plot(timesteps, pred_series, label="T1 prediction", linestyle="--")
    plt.scatter(timesteps[~mask], pred_series[~mask], color="red", label="Missing input", s=10)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    if savepath:
        _ensure_output_dir(Path(savepath))
        plt.savefig(savepath)
    plt.close()


def plot_multivariate_sample(
    gt_data: np.ndarray,
    pred_data: np.ndarray,
    mask: np.ndarray,
    variable_ids: List[int],
    sample_id: int,
    savepath: Optional[Path] = None,
) -> None:
    """Plot multiple variables for a single sample.

    The function is designed to be model-agnostic so that additional model
    predictions can be overlaid later.
    """

    if not _matplotlib_available():
        return
    n_vars = len(variable_ids)
    fig, axes = plt.subplots(n_vars, 1, figsize=(8, 3 * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]

    for ax, vid in zip(axes, variable_ids):
        gt_series = gt_data[sample_id, :, vid]
        pred_series = pred_data[sample_id, :, vid]
        mask_series = mask[sample_id, :, vid]
        timesteps = np.arange(gt_series.shape[0])
        ax.plot(timesteps, gt_series, label="Ground truth", linewidth=2)
        ax.plot(timesteps, pred_series, linestyle="--", label="T1 prediction")
        ax.scatter(timesteps[~mask_series], pred_series[~mask_series], color="red", s=10, label="Missing input")
        ax.set_title(f"Variable {vid}")
        ax.set_ylabel("Value")
    axes[-1].set_xlabel("Time")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    if savepath:
        _ensure_output_dir(Path(savepath))
        plt.savefig(savepath)
    plt.close()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def compute_summary_table(
    var_groups: Mapping[str, List[int]],
    missing_df: pd.DataFrame,
    group_mse: Mapping[str, Mapping[float, float]],
    target_ratio: float = 0.5,
) -> pd.DataFrame:
    """Assemble a summary table for quick reporting."""

    _ensure_core_dependencies()
    rows = []
    for group, vids in var_groups.items():
        natural_missing = missing_df.loc[missing_df["variable_id"].isin(vids), "missing_rate"].mean()
        t1_mse = group_mse.get(group, {}).get(target_ratio, np.nan)
        rows.append(
            {
                "Variable Group": group,
                "Natural Missing (avg)": natural_missing,
                "T1 MSE at target_ratio": t1_mse,
                "Best Baseline MSE": np.nan,
                "Improvement": np.nan,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    zip_parts_pattern = "physio_part_*"
    zip_path = base_dir / "physionet2012_T1_results.zip"
    extracted_dir = base_dir / "physionet2012_T1_results"
    figs_dir = base_dir / "figures"

    # Dependency status influences downstream analytics, but reconstruction and
    # structure inspection should still run even when numpy/pandas are missing.
    dependencies_ok = _core_dependencies_available()

    # Reconstruct the ZIP archive.
    try:
        reconstruct_zip_from_parts(zip_parts_pattern, zip_path)
    except FileNotFoundError as exc:
        print(f"[WARN] Could not reconstruct ZIP: {exc}")
        return

    if is_git_lfs_pointer(zip_path):
        print("[WARN] The reconstructed ZIP appears to be a Git LFS pointer. "
              "Please fetch the real data before running analyses.")
        return

    # Explore ZIP contents.
    try:
        contents = list_zip_contents(zip_path)
        print("[INFO] ZIP contents:")
        for name in contents:
            print(f"  - {name}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Unable to read ZIP contents: {exc}")
        return

    # Extract files.
    try:
        extract_zip(zip_path, extracted_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to extract ZIP: {exc}")
        return

    # Print directory tree.
    print("[INFO] Extracted directory tree:")
    print(describe_directory_tree(extracted_dir, max_depth=4))

    # Load metadata and metrics (requires numpy/pandas).
    data_info_path = extracted_dir / "data_info.json"
    metrics_path_root = extracted_dir
    ground_truth_dir = extracted_dir / "ground_truth"

    data_info = None
    metrics_summary = None
    gt_data = None
    gt_mask = None
    gt_metadata: dict = {}

    if not dependencies_ok:
        print(
            "[WARN] numpy/pandas not available; skipping metric loading and analyses. "
            "Install the dependencies to enable full evaluation."
        )
        return

    if data_info_path.exists():
        data_info = load_data_info(data_info_path)
        print("[INFO] Loaded data_info.json")
    else:
        print("[WARN] data_info.json not found.")

    try:
        metrics_summary = load_metrics_summary(metrics_path_root)
        print("[INFO] Loaded metrics summary with columns:", metrics_summary.columns.tolist())
    except FileNotFoundError as exc:
        print(f"[WARN] Metrics file not found: {exc}")

    try:
        gt_data, gt_mask, gt_metadata = load_ground_truth(ground_truth_dir)
        print(f"[INFO] Ground truth shape: {gt_data.shape}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not load ground truth: {exc}")

    t1_predictions = load_t1_predictions(extracted_dir)

    # If we have everything needed, compute analyses.
    if gt_data is not None and metrics_summary is not None:
        # Compute missingness statistics.
        missing_df = compute_variable_missing_rates(gt_data, gt_mask)
        var_groups = group_variables_by_missing_rate(missing_df)

        # Build model results from metrics_summary (assumes ratios encoded as mrXX columns).
        model_results: Dict[str, Dict[float, float]] = {"T1": {}}
        for col in metrics_summary.columns:
            match = re.match(r"mr(\d+)", col)
            if match:
                pct = int(match.group(1))
                model_results["T1"][pct / 100.0] = float(metrics_summary[col].mean())

        # Plot overall degradation.
        overall_path = figs_dir / "figure_overall_degradation.png"
        _attempt_plot("overall degradation plot", plot_overall_degradation, model_results, savepath=overall_path)

        # Compute group MSE using predictions when available.
        group_mse = compute_group_mse_by_ratio(gt_data, t1_predictions, gt_mask, var_groups)
        group_path = figs_dir / "figure_variable_groups.png"
        _attempt_plot("variable-group degradation plot", plot_variable_group_degradation, group_mse, savepath=group_path)

        # Extreme case plot placeholder: compute per-variable MSE by ratio.
        mse_by_var_and_ratio: Dict[float, Dict[int, float]] = {}
        for ratio, preds in t1_predictions.items():
            if preds is None:
                continue
            pred_data, pred_mask, _ = _infer_data_and_mask(preds)
            aligned_mask = gt_mask & pred_mask
            per_var = compute_mse_per_variable(gt_data, pred_data, aligned_mask)
            mse_by_var_and_ratio[ratio] = dict(zip(per_var["variable_id"], per_var["mse"]))

        extreme_vars = var_groups.get("extreme", [])
        if extreme_vars:
            extreme_path = figs_dir / "figure_extreme_vars.png"
            _attempt_plot("extreme missingness plot", plot_extreme_case_mse, extreme_vars, mse_by_var_and_ratio, savepath=extreme_path)
        else:
            print("[INFO] No extreme-missing variables detected; skipping extreme plot.")

        # Example cherry-picked plots (placeholders).
        sample_id = 0
        if t1_predictions:
            # Use the smallest missing ratio available for visualization.
            first_ratio = sorted(t1_predictions.keys())[0]
            preds = t1_predictions[first_ratio]
            if preds is not None:
                pred_data, pred_mask, _ = _infer_data_and_mask(preds)
                # Pick representative variables.
                low_vars = var_groups.get("low", [0])
                high_vars = var_groups.get("high", []) or var_groups.get("extreme", []) or [0]
                if low_vars:
                    low_path = figs_dir / "figure_low_missing_variable.png"
                    _attempt_plot(
                        "low-missing variable plot",
                        plot_single_variable_timeseries,
                        gt_data[sample_id, :, low_vars[0]],
                        pred_data[sample_id, :, low_vars[0]],
                        gt_mask[sample_id, :, low_vars[0]],
                        title=f"Sample {sample_id} - Variable {low_vars[0]}",
                        savepath=low_path,
                    )
                if high_vars:
                    high_path = figs_dir / "figure_high_missing_variable.png"
                    _attempt_plot(
                        "high/extreme-missing variable plot",
                        plot_single_variable_timeseries,
                        gt_data[sample_id, :, high_vars[0]],
                        pred_data[sample_id, :, high_vars[0]],
                        gt_mask[sample_id, :, high_vars[0]],
                        title=f"Sample {sample_id} - Variable {high_vars[0]}",
                        savepath=high_path,
                    )
                multi_path = figs_dir / "figure_multivariate_sample.png"
                _attempt_plot(
                    "multivariate sample plot",
                    plot_multivariate_sample,
                    gt_data,
                    pred_data,
                    gt_mask,
                    variable_ids=list(set(low_vars[:1] + high_vars[:1])),
                    sample_id=sample_id,
                    savepath=multi_path,
                )
            
        summary_table = compute_summary_table(var_groups, missing_df, group_mse)
        print("\n[INFO] Summary table:")
        print(summary_table)
    else:
        print("[INFO] Skipping downstream analyses because required data is missing.")


if __name__ == "__main__":
    main()
