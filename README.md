# robust-ocean-sensor-placement
Robust differentiable sensor placement for ocean-state reconstruction with learned non-stationary covariance geometry.

## Experiment B: first results

`experiment_B_25days_final.py` (commit `215d236`) was run unchanged on all 25 days × 19 conditions:
475 runs on AMD MI210 GPUs, about 325 GPU-hours in total. Each method learns sensor placement and
covariance from an imperfect 10-member forecast ensemble. It is then scored on the true NATL60 field.
Full tables, files and caveats are in [data/experiment_B/README.md](data/experiment_B/README.md).

RMSE on the true field, in cm, averaged over the 25 days:

| Forecast | Random | Isotropic | A2 | A3 | A3 gain over Isotropic |
|---|---|---|---|---|---|
| clean | 18.3 | 11.2 | 10.9 | 7.4 | 34.3%, 25/25 days |
| all 18 error conditions | 18.3 | 11.9 | 11.7 | 8.0 | 33.3%, 448/450 |
| strong errors only | 18.3 | 12.5 | 12.3 | 8.8 | 30.4%, 149/150 |
| smoothing, strong | 18.3 | 15.5 | 15.4 | 13.3 | 14.5%, 24/25 |

What to look at first:

- A3 beats Isotropic on almost every day and condition. Its gain shrinks only a little as forecast errors get stronger.
- Over-smoothed forecasts are the weak spot: A3's gain falls to 19% (medium) and 15% (strong).
- A2 adds little, about 6% of the Isotropic-to-A3 gain. The non-stationary warp does the work.
- A3 barely moves the Isotropic placement, and both use about 67 sensors. The gain therefore comes
  from the covariance, not from where the sensors go.
- Training on several perturbed forecasts beats training on the clean truth. For example, A3 gets
  6.7 cm under medium amplitude error against 7.4 cm on clean. The clean ensemble is 10 identical copies.
- There is only one model seed and one perturbation seed, and the displacement fields are the same every day.

## Possible next steps

1. **Placement vs covariance swap:** score the A3 sensors with the Isotropic covariance, and the
   reverse, from `experiment_B_states.pt` without retraining. This shows whether the learned placement matters.
2. **Fairer Random baseline:** random sensors with a learned covariance instead of the fixed length
   scale of 25, also without retraining.
3. **Seeds:** rerun the clean and medium conditions with 2–3 more model and perturbation seeds and new displacement fields.
4. **Clean vs perturbed:** find out why perturbed training beats clean training. Candidates are
   ensemble diversity, checkpoint selection on 16 draws, and training length.
5. **Smoothing:** check what the covariance learns from over-smoothed forecasts, and whether that can be corrected.
6. **Realism:** sweep the sensor budget (`RATE` is fixed at 0.999), and test a real forecast error,
   for example day t−k used as the forecast for day t.
7. **Experiment A:** the script refers to it, but it is not in the repository. Add it so both
   experiments can be compared on the same hardware.
