"""Report Experiment B results from a merged output folder.

    python src/report_b.py outputs/full/merged

For each error type and severity (mean over days, RMSE on the clean truth in metres):
  - Random, Isotropic, A2 and A3 RMSE, and EV of Isotropic and A3;
  - A3 vs Isotropic: mean per-day relative RMSE gain, days A3 wins, Wilcoxon signed-rank p;
  - A2 share: part of the Isotropic-to-A3 RMSE gain already reached by A2;
  - A2 L ratio: mean max(L1, L2) / min(L1, L2), how anisotropic A2 became.
Pooled rows follow: each severity over the six error types, and all 18 error conditions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ERRORS = ["clean", "deformation", "amplitude", "large_scale_bias", "noise", "smoothing", "combined"]
SEVERITIES = ["clean", "mild", "medium", "strong"]
KEYS = ["error_type", "severity", "day_index"]


def compare(rmse, ev, ratio, forecast):
    iso, a2, a3, rnd = (rmse[m].to_numpy() for m in ("Isotropic", "A2", "A3", "Random"))
    try:
        p = wilcoxon(a3, iso).pvalue
    except ValueError:  # all paired differences are zero
        p = float("nan")
    total = iso.mean() - a3.mean()
    return {
        "forecast err": forecast,
        "Random": rnd.mean(),
        "Iso": iso.mean(),
        "A2": a2.mean(),
        "A3": a3.mean(),
        "EV Iso": ev["Isotropic"].mean(),
        "EV A3": ev["A3"].mean(),
        "A3 vs Iso %": 100 * ((iso - a3) / iso).mean(),
        "A3 wins": f"{int((a3 < iso).sum())}/{len(iso)}",
        "p": p,
        "A2 share %": 100 * (iso.mean() - a2.mean()) / total if total else float("nan"),
        "A2 L ratio": ratio.mean(),
        "Iso vs Random %": 100 * ((rnd - iso) / rnd).mean(),
    }


def main(folder):
    folder = Path(folder)
    daily = pd.read_csv(folder / "experiment_B_daily.csv")
    a2 = pd.read_csv(folder / "a2_truth.csv")
    rows = pd.concat([daily, a2[daily.columns]], ignore_index=True)
    rmse = rows.pivot_table(index=KEYS, columns="Method", values="RMSE (m)")
    ev = rows.pivot_table(index=KEYS, columns="Method", values="EV")
    ratio = (np.maximum(a2["L1"], a2["L2"]) / np.minimum(a2["L1"], a2["L2"])).set_axis(
        pd.MultiIndex.from_frame(a2[KEYS])
    ).reindex(rmse.index)
    forecast = daily.groupby(KEYS)["forecast_RMSE_norm"].first().reindex(rmse.index)

    def row(label_error, label_severity, mask):
        out = {"error_type": label_error, "severity": label_severity}
        out.update(compare(rmse[mask], ev[mask], ratio[mask], forecast[mask].mean()))
        return out

    error = rmse.index.get_level_values("error_type")
    severity = rmse.index.get_level_values("severity")
    table = [
        row(e, s, (error == e) & (severity == s))
        for e in ERRORS for s in SEVERITIES
        if ((error == e) & (severity == s)).any()
    ]
    table += [row("all errors", s, (error != "clean") & (severity == s)) for s in SEVERITIES[1:]]
    table.append(row("all errors", "all", error != "clean"))

    days = rmse.index.get_level_values("day_index").nunique()
    print(f"{folder}: {days} days, {len(rmse)} day-conditions; RMSE in metres on the clean truth")
    sensors = rows.groupby("Method")["Sensors"].mean().round(1).to_dict()
    print(f"mean sensors per method: {sensors}")
    pct = "{:.1f}".format
    formats = {
        **{c: "{:.4f}".format for c in ["forecast err", "Random", "Iso", "A2", "A3", "EV Iso", "EV A3"]},
        "A3 vs Iso %": pct, "A2 share %": pct, "Iso vs Random %": pct,
        "p": "{:.1e}".format, "A2 L ratio": "{:.2f}".format,
    }
    print(pd.DataFrame(table).to_string(index=False, formatters=formats))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
