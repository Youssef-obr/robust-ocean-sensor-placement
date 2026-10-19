# Experiment B results

A budget of about 78 sensors (0.1% of a 280×280 Gulf Stream grid of NATL60 sea surface height)
is placed, and the full map is reconstructed from them by optimal interpolation. Placement and covariance are
learned from an imperfect forecast ensemble, then scored on the true field. A3, the learned
non-stationary covariance, reconstructs the truth with about 33% lower RMSE than Isotropic. It wins
on 448 of the 450 day × forecast-error pairs.

## What was run

- Script: `experiment_B_25days_final.py` at commit `215d236`, unchanged.
- Days: the 25 NATL60 days of `time[::15]`.
- Conditions: 19 per day, clean plus 6 forecast-error types × 3 severities. Each forecast is a 10-member ensemble.
- Training: for each condition, Isotropic, then A2, then A3, on the forecast ensemble.
- Scoring: on the clean truth, with 100 fixed-seed sensor draws.
- Random baseline: untrained placement at the same rate, with a fixed length scale of 25.
- Hardware: AMD Instinct MI210 (ROCm 6.4, torch 2.8.0+rocm6.4), one GPU per day, in October 2026.
- Cost: about 41 minutes per condition, 475 conditions, about 325 GPU-hours in total.
- `src/run_b.py` ran the script. It adds timing lines and `a2_truth.csv`, which scores A2 on the truth after A3 is trained, with the same masks as the other methods.
  - Experiment B's own outputs are unchanged. A check run gave a byte-identical `experiment_B_daily.csv` with and without it.
- `src/run_b.py merge` joined the 25 day runs. `src/report_b.py` wrote `report.txt`.

## Files

| File | Content |
|---|---|
| `experiment_B_daily.csv` | One row per day, condition and method (A3, Isotropic, Random): RMSE (m), EV, sensors, and forecast RMSE (normalized) |
| `experiment_B_summary.csv` | The same metrics averaged over the 25 days, per condition and method |
| `a2_truth.csv` | A2 scored on the truth, one row per day and condition, with its learned `L1`, `L2` and `theta` |
| `a2_summary.csv` | `a2_truth.csv` averaged over the 25 days |
| `report.txt` | Per-condition table: mean RMSE of all four methods, A3 vs Isotropic gain, days won, Wilcoxon p, share of the gain reached by A2 |

## Main numbers

RMSE on the true field in cm, averaged over 25 days. "A3 gain" is the mean per-day RMSE reduction against Isotropic.

| Forecast error | Random | Isotropic | A2 | A3 | A3 gain | A3 wins |
|---|---|---|---|---|---|---|
| none (clean) | 18.3 | 11.2 | 10.9 | 7.4 | 34.3% | 25/25 |
| all 18 error conditions | 18.3 | 11.9 | 11.7 | 8.0 | 33.3% | 448/450 |
| mild (6 error types) | 18.3 | 11.4 | 11.1 | 7.3 | 36.0% | 150/150 |
| medium | 18.3 | 11.8 | 11.6 | 7.9 | 33.7% | 149/150 |
| strong | 18.3 | 12.5 | 12.3 | 8.8 | 30.4% | 149/150 |
| smoothing, medium | 18.3 | 13.3 | 13.0 | 10.8 | 19.0% | 24/25 |
| smoothing, strong | 18.3 | 15.5 | 15.4 | 13.3 | 14.5% | 24/25 |

- Over-smoothed forecasts are the only error type that strongly reduces A3's advantage.
  Amplitude and large-scale bias errors barely affect any method.
- A2 reaches only about 6% of the Isotropic-to-A3 gain (4.7% to 14% across conditions). Its two
  length scales differ by a factor of about 1.2. Most of A3's gain comes from the non-stationary warp.
- The trained methods use about 67 sensors on average, and Random uses about 78.

## Caveats

- There is one model seed and one perturbation seed, and the displacement fields are the same every day.
- The Random baseline uses a fixed length scale (25), not a learned one.
- A3 starts from the Isotropic placement and barely moves it, so both use nearly the same sensors.
  The gain therefore comes from the covariance. A placement/covariance swap has not been run.
- Several perturbed conditions reconstruct the truth better than the clean one. For example, A3 gets
  6.7 cm with medium amplitude error against 7.4 cm on clean. The clean "ensemble" is 10 identical copies.
- Numbers come from AMD GPUs (ROCm). Runs on NVIDIA GPUs can differ slightly.

## Not included

`experiment_B_states.pt` (308 MB) holds, per condition, the effective sensor logits and covariance
parameters of Isotropic and A3, and the A3 warp weights. It exceeds GitHub's 100 MB file limit and
is kept outside the repository.

## Reproduce

```bash
GS_OUTPUT_DIR=out/day-00 GS_START_DAY=0 GS_END_DAY=1 python src/run_b.py run   # one day; repeat for days 0-24
python src/run_b.py merge out/merged out/day-*
python src/report_b.py out/merged
```
