"""Run Experiment B of robust-ocean-sensor-placement without editing it.

    python src/run_b.py run                  # one shard; settings come from the environment
    python src/run_b.py merge OUT SHARD...   # join shard outputs into the three final files

The experiment script is imported unchanged. This wrapper only adds:
  - B_CONDITIONS: optional comma list of error/severity pairs, e.g. "clean/clean,deformation/medium";
  - a header with package versions and the script hash, and per-stage wall times (lines "TIMING ...");
  - a2_truth.csv: each condition's A2 (the bridge model the script trains, then drops) scored on the
    clean truth like Isotropic and A3, plus its L1, L2 and theta. The three Experiment B files are unchanged.
GS_OUTPUT_DIR is required, so nothing is written inside the experiment repository.
"""

import hashlib
import importlib.util
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("B_REPO", Path(__file__).resolve().parents[1]))
SCRIPT = REPO / "experiment_B_25days_final.py"
TIMED = [
    "build_all_forecast_ensembles",
    "train_isotropic",
    "train_a2_from_isotropic",
    "train_a3_from_a2",
    "evaluate_model_on_truth",
    "evaluate_random_on_truth",
]
A2_CSV = "a2_truth.csv"
A2_SUMMARY_CSV = "a2_summary.csv"
A2_PARAMS = ["L1", "L2", "theta"]


def load_experiment():
    if not os.environ.get("GS_OUTPUT_DIR"):
        sys.exit("GS_OUTPUT_DIR is not set; refusing to write inside the experiment repository.")
    spec = importlib.util.spec_from_file_location("experiment_b", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_commit():
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def print_header(B):
    import netCDF4
    import numpy
    import pandas
    import torch
    import xarray

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"host={platform.node()} python={platform.python_version()} gpu={gpu}")
    print(
        f"torch={torch.__version__} hip={torch.version.hip} numpy={numpy.__version__} "
        f"pandas={pandas.__version__} xarray={xarray.__version__} netCDF4={netCDF4.__version__}"
    )
    print(f"script={SCRIPT.name} sha256={hashlib.sha256(SCRIPT.read_bytes()).hexdigest()[:12]} commit={git_commit()}")
    print(f"data={B.DATA_PATH} output={B.OUTPUT_DIR}")
    overrides = {k: v for k, v in sorted(os.environ.items()) if k.startswith(("B_", "GS_", "NATL60_"))}
    print(f"env={overrides}")
    print(f"conditions={len(B.CONDITIONS)}", flush=True)


def timed(name, fn):
    import torch

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"TIMING {name} {time.perf_counter() - start:.1f}s", flush=True)
        return result

    return wrapper


def select_conditions(B, spec):
    wanted = [tuple(item.strip().split("/")) for item in spec.split(",") if item.strip()]
    unknown = [w for w in wanted if w not in B.CONDITIONS]
    if unknown:
        sys.exit(f"Unknown conditions {unknown}; valid: {B.CONDITIONS}")
    B.CONDITIONS = [c for c in B.CONDITIONS if c in wanted]


def add_a2_log(B):
    """Write one A2 row per condition to GS_OUTPUT_DIR/a2_truth.csv.

    The row is written when the script exports the condition, after A3 is trained, with the script's own
    evaluate_model_on_truth (same fixed-seed masks as Isotropic and A3, no gradients, no global RNG).
    A3 training and the three Experiment B files are therefore unchanged.
    """
    import pandas as pd

    path = B.OUTPUT_DIR / A2_CSV
    columns = B.DAILY_COLUMNS + A2_PARAMS
    table = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)
    state = {}
    random_fn, train_fn, export_fn = (
        B.evaluate_random_on_truth, B.train_a2_from_isotropic, B.export_condition_state
    )

    def evaluate_random_on_truth(clean_norm, clean_std_m, day_index, lat, lon):
        # Called once per day before its conditions: the clean truth the methods are scored on.
        state["truth"] = (day_index, clean_norm)
        return random_fn(clean_norm, clean_std_m, day_index, lat, lon)

    def train_a2_from_isotropic(*args, **kwargs):
        state["a2"] = train_fn(*args, **kwargs)
        return state["a2"]

    def export_condition_state(iso_model, a3_model, **kw):
        nonlocal table
        day_index, clean_norm = state["truth"]
        if day_index != kw["day_index"]:
            raise RuntimeError(f"A2 log: truth of day {day_index}, condition of day {kw['day_index']}")
        a2 = state.pop("a2")
        metrics = B.evaluate_model_on_truth(a2, clean_norm, kw["std_tgt"], day_index)
        L1, L2, theta = (float(v.detach().cpu()) for v in a2.covariance_params())
        row = {
            "day_index": day_index, "date": kw["date_str"], "error_type": kw["error_type"],
            "severity": kw["severity"], "forecast_RMSE_norm": kw["forecast_rmse"], "Method": "A2",
            **metrics, "L1": L1, "L2": L2, "theta": theta,
        }
        same = (
            (table["day_index"] == day_index)
            & (table["error_type"] == kw["error_type"])
            & (table["severity"] == kw["severity"])
        )
        table = pd.DataFrame(table.loc[~same].to_dict("records") + [row], columns=columns)
        B.atomic_write_csv(table, path)
        print(
            f"    CLEAN truth eval A2:       RMSE={metrics['RMSE (m)']:.6f} m | EV={metrics['EV']:.6f} | "
            f"sensors={metrics['Sensors']:.2f} | L1={L1:.3f} L2={L2:.3f} theta={theta:.3f}",
            flush=True,
        )
        return export_fn(iso_model, a3_model, **kw)

    B.evaluate_random_on_truth = evaluate_random_on_truth
    B.train_a2_from_isotropic = train_a2_from_isotropic
    B.export_condition_state = export_condition_state


def run():
    B = load_experiment()
    if os.environ.get("B_CONDITIONS"):
        select_conditions(B, os.environ["B_CONDITIONS"])
    for name in TIMED:
        setattr(B, name, timed(name, getattr(B, name)))
    add_a2_log(B)
    print_header(B)
    start = time.perf_counter()
    B.main()
    print(f"TIMING total {time.perf_counter() - start:.1f}s", flush=True)


def merge(out_dir, shard_dirs):
    os.environ["GS_OUTPUT_DIR"] = out_dir
    B = load_experiment()
    import pandas as pd
    import torch

    frames, bundle = [], None
    for shard in map(Path, shard_dirs):
        frames.append(pd.read_csv(shard / B.DAILY_CSV.name))
        part = torch.load(shard / B.STATES_PT.name, map_location="cpu", weights_only=False)
        if bundle is None:
            bundle = part
            continue
        if part["config"] != bundle["config"]:
            sys.exit(f"{shard}: config differs from the first shard")
        overlap = set(part["conditions"]) & set(bundle["conditions"])
        if overlap:
            sys.exit(f"{shard}: {len(overlap)} conditions already merged, e.g. {sorted(overlap)[0]}")
        bundle["conditions"].update(part["conditions"])

    daily = pd.concat(frames, ignore_index=True)[B.DAILY_COLUMNS]
    keys = ["day_index", "error_type", "severity", "Method"]
    if daily.duplicated(keys).any():
        sys.exit(f"duplicate rows: {daily[daily.duplicated(keys)][keys].head().to_dict('records')}")
    daily = sort_rows(B, daily, by_day=True)

    expected = 25 * len(B.CONDITIONS)
    print(f"merged {len(shard_dirs)} shards: {len(bundle['conditions'])}/{expected} conditions, {len(daily)} rows")
    B.atomic_write_csv(daily, B.DAILY_CSV)
    B.write_summary(daily)
    B.atomic_torch_save(bundle, B.STATES_PT)
    print(f"wrote {B.DAILY_CSV}, {B.SUMMARY_CSV}, {B.STATES_PT}")
    merge_a2(B, shard_dirs, set(bundle["conditions"]))


def sort_rows(B, df, by_day):
    order = df.assign(
        _e=df["error_type"].map(B.ERROR_ORDER),
        _s=df["severity"].map(B.SEVERITY_ORDER),
        _m=df["Method"].map({**B.METHOD_ORDER, "A2": -1}),
    )
    by = (["day_index"] if by_day else []) + ["_e", "_s", "_m"]
    return order.sort_values(by).drop(columns=["_e", "_s", "_m"])


def merge_a2(B, shard_dirs, conditions):
    import pandas as pd

    parts = [pd.read_csv(Path(s) / A2_CSV) for s in shard_dirs if (Path(s) / A2_CSV).exists()]
    if not parts:
        print(f"no {A2_CSV} in the shards")
        return
    a2 = pd.concat(parts, ignore_index=True)
    keys = ["day_index", "error_type", "severity"]
    if a2.duplicated(keys).any():
        sys.exit(f"duplicate A2 rows: {a2[a2.duplicated(keys)][keys].head().to_dict('records')}")
    logged = {B.state_key(d, e, s) for d, e, s in a2[keys].itertuples(index=False)}
    print(f"A2 rows: {len(logged & conditions)}/{len(conditions)} merged conditions, {len(logged - conditions)} extra")
    a2 = sort_rows(B, a2, by_day=True)
    B.atomic_write_csv(a2, B.OUTPUT_DIR / A2_CSV)
    metrics = ["forecast_RMSE_norm", "RMSE (m)", "EV", "Sensors"] + A2_PARAMS
    summary = a2.groupby(["error_type", "severity", "Method"], as_index=False)[metrics].mean()
    B.atomic_write_csv(sort_rows(B, summary, by_day=False), B.OUTPUT_DIR / A2_SUMMARY_CSV)
    print(f"wrote {B.OUTPUT_DIR / A2_CSV}, {B.OUTPUT_DIR / A2_SUMMARY_CSV}")


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        run()
    elif len(sys.argv) >= 4 and sys.argv[1] == "merge":
        merge(sys.argv[2], sys.argv[3:])
    else:
        sys.exit(__doc__)
