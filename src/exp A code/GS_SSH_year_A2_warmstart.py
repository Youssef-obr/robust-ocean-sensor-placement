import os
import copy
from pathlib import Path
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import xarray as xr
from torch.amp import autocast, GradScaler


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def first_existing(*paths):
    for p in paths:
        if p is not None and Path(p).exists():
            return str(Path(p))
    return str(paths[0]) if paths else None


DATA_PATH = os.environ.get(
    "NATL60_PATH",
    first_existing(
        SCRIPT_DIR / "NATL60_GulfStream_ssh.nc",
        "/kaggle/input/datasets/youssefoubari/natl60-gulf-stream-ssh/NATL60_GulfStream_ssh.nc",
        "C:/Users/gerv9/Desktop/gumbell/data/NATL60-CJM165-daily-reg-1_20.nc",
    ),
)

BASELINE_RESULTS_PATH = os.environ.get(
    "BASELINE_RESULTS_PATH",
    str(SCRIPT_DIR / "results_natl60.nc"),
)
BASELINE_LOGITS_PATH = os.environ.get(
    "BASELINE_LOGITS_PATH",
    str(SCRIPT_DIR / "logits_natl60.nc"),
)

OUTPUT_DIR = Path(
    os.environ.get(
        "GS_OUTPUT_DIR",
        str(SCRIPT_DIR / "outputs_A2_warmstart"),
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# We now fine-tune from the trained isotropic solution instead of retraining
# A2 from scratch.
STAGE1_ITERS = int(os.environ.get("A2_STAGE1_ITERS", "150"))  # covariance only
STAGE2_ITERS = int(os.environ.get("A2_STAGE2_ITERS", "250"))  # covariance + logits
TOTAL_ITERS = STAGE1_ITERS + STAGE2_ITERS

# Same effective number of stochastic draws per optimizer update as the
# validated memory-safe baseline.
N_DRAW = 1
ACCUMULATION_STEPS = int(os.environ.get("A2_ACCUMULATION_STEPS", "60"))

# Smaller fine-tuning learning rates than the from-scratch run.
COV_LR = float(os.environ.get("A2_COV_LR", "0.02"))
LOGITS_LR = float(os.environ.get("A2_LOGITS_LR", "0.01"))

RATE = 0.999
BUDGET_OBS = 1.0 - RATE              # 0.001 = 0.1 %
MAX_WEIGHT = 75.0                    # active immediately during joint fine-tuning
START_TEMP = 15.0
MIN_TEMP = 4.0
OI_CHUNK_SIZE = int(os.environ.get("A2_OI_CHUNK_SIZE", "5000"))

# Checkpoint selection uses fixed hard Gumbel masks so every candidate is
# compared on exactly the same stochastic realizations.
EVAL_EVERY = int(os.environ.get("A2_EVAL_EVERY", "50"))
EVAL_DRAWS = int(os.environ.get("A2_EVAL_DRAWS", "16"))
EVAL_SEED = int(os.environ.get("A2_EVAL_SEED", "271828"))

# A candidate is only eligible if its deterministic expected observation
# fraction does not exceed the nominal paper budget. If the reconstructed
# baseline itself is slightly above the budget, we allow exactly that baseline
# level so the initial checkpoint is always eligible.
MIN_REL_IMPROVEMENT = float(os.environ.get("A2_MIN_REL_IMPROVEMENT", "0.0"))

# Optional chunking / smoke testing without modifying the file.
START_DAY = int(os.environ.get("GS_START_DAY", "0"))     # zero-based, inclusive
END_DAY = int(os.environ.get("GS_END_DAY", "25"))       # zero-based, exclusive


# =============================================================================
# Rotated anisotropic Gaussian / RBF covariance
# =============================================================================

def _as_tensor(value, *, device, dtype=torch.float32):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, dtype=dtype, device=device)


def _anisotropic_rbf_covariance(x1, x2, L1, L2, theta, sigma_f):
    """
    Rotated anisotropic RBF covariance.

    Coordinates are [x, y].

    delta = r_i - r_j
    d1 =  cos(theta) * dx + sin(theta) * dy
    d2 = -sin(theta) * dx + cos(theta) * dy

    C = sigma_f^2 * exp[-0.5 * (d1^2/L1^2 + d2^2/L2^2)]

    If L1 == L2, this is exactly the isotropic RBF kernel and theta has no
    effect.
    """
    x1 = x1.unsqueeze(1)
    x2 = x2.unsqueeze(0)

    delta = x1 - x2
    dx = delta[..., 0]
    dy = delta[..., 1]

    c = torch.cos(theta)
    s = torch.sin(theta)

    d1 = c * dx + s * dy
    d2 = -s * dx + c * dy

    anisotropic_sqdist = (d1 / L1) ** 2 + (d2 / L2) ** 2
    return sigma_f**2 * torch.exp(-0.5 * anisotropic_sqdist)


# =============================================================================
# Memory-safe optimal interpolation
# =============================================================================

def optimal_interpolation_chunk(
    data_with_nans,
    L1,
    L2,
    theta,
    sigma_f=1.0,
    sigma_n=0.1,
    chunk_size=5000,
):
    device_ = data_with_nans.device

    observed_indices = ~torch.isnan(data_with_nans)
    missing_indices = torch.isnan(data_with_nans)

    observed_positions = torch.nonzero(observed_indices, as_tuple=False).float()
    missing_positions = torch.nonzero(missing_indices, as_tuple=False).float()

    observed_values = data_with_nans[observed_indices].float().to(device_)

    # [x, y], identical convention to the original implementation.
    observed_coords = observed_positions[:, [1, 0]].to(device_)
    missing_coords = missing_positions[:, [1, 0]].to(device_)

    L1 = _as_tensor(L1, device=device_)
    L2 = _as_tensor(L2, device=device_)
    theta = _as_tensor(theta, device=device_)
    sigma_f = _as_tensor(sigma_f, device=device_)
    sigma_n = _as_tensor(sigma_n, device=device_)

    def cov_func(x1, x2):
        return _anisotropic_rbf_covariance(
            x1, x2, L1=L1, L2=L2, theta=theta, sigma_f=sigma_f
        )

    K_oo = cov_func(observed_coords, observed_coords)
    eye = torch.eye(K_oo.size(0), device=device_, dtype=K_oo.dtype)
    K_oo = K_oo + sigma_n**2 * eye

    L_chol = torch.linalg.cholesky(K_oo + 1e-6 * eye)
    alpha_oi = torch.cholesky_solve(observed_values.unsqueeze(1), L_chol)

    num_missing = missing_coords.size(0)
    pred_list = []

    for start in range(0, num_missing, chunk_size):
        end = min(start + chunk_size, num_missing)
        K_s_chunk = cov_func(
            observed_coords,
            missing_coords[start:end],
        )
        pred_chunk = K_s_chunk.t().matmul(alpha_oi).squeeze(1)
        pred_list.append(pred_chunk)

        del K_s_chunk, pred_chunk

    if len(pred_list) > 0:
        mean_pred = torch.cat(pred_list, dim=0)
    else:
        mean_pred = torch.empty(0, device=device_, dtype=torch.float32)

    reconstructed_data = data_with_nans.clone()
    if mean_pred.numel() > 0:
        reconstructed_data[missing_indices] = mean_pred.to(reconstructed_data.dtype)

    return reconstructed_data


# =============================================================================
# Gumbel sensor-placement model
# =============================================================================

def inverse_softplus(y):
    """
    Stable inverse of softplus for positive y.
    softplus(inverse_softplus(y)) == y up to floating point precision.
    """
    y = torch.as_tensor(y, dtype=torch.float32)
    return y + torch.log(-torch.expm1(-y))


class GSModel(nn.Module):
    def __init__(self, time, lat, lon, device):
        super().__init__()

        self.logits = nn.Parameter(torch.zeros((2, time, lat, lon)))
        self.time_window = time
        self.lat = lat
        self.lon = lon
        self.device = device

        # Values are overwritten by initialize_from_baseline().
        self.raw_L1 = nn.Parameter(torch.tensor(25.0, dtype=torch.float32))
        self.raw_L2 = nn.Parameter(torch.tensor(25.0, dtype=torch.float32))
        self.theta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def covariance_params(self):
        eps = 1e-6
        L1 = F.softplus(self.raw_L1) + eps
        L2 = F.softplus(self.raw_L2) + eps
        return L1, L2, self.theta

    @torch.no_grad()
    def initialize_from_baseline(self, baseline_effective_logit, baseline_L):
        """
        Reconstruct the saved isotropic baseline convention:
          class-0 logit = 0
          class-1 logit = saved baseline logit

        Then set L1 = L2 = L_baseline and theta = 0.

        This makes the A2 covariance exactly isotropic at initialization.
        """
        self.logits.zero_()

        t = torch.as_tensor(
            baseline_effective_logit,
            dtype=self.logits.dtype,
            device=self.logits.device,
        )
        if t.ndim == 2:
            t = t.unsqueeze(0)

        if tuple(t.shape) != tuple(self.logits[1].shape):
            raise ValueError(
                f"Baseline logits shape {tuple(t.shape)} does not match "
                f"model shape {tuple(self.logits[1].shape)}."
            )

        self.logits[1].copy_(t)

        target_L = torch.tensor(
            float(baseline_L) - 1e-6,
            device=self.raw_L1.device,
            dtype=self.raw_L1.dtype,
        ).clamp_min(1e-6)

        raw = inverse_softplus(target_L).to(self.raw_L1.device)
        self.raw_L1.copy_(raw)
        self.raw_L2.copy_(raw)
        self.theta.zero_()

    def effective_logit(self):
        """
        Binary logit difference. Saving this quantity is sufficient to recreate
        exactly the same categorical probabilities with class-0 logit fixed to 0.
        """
        return self.logits[1] - self.logits[0]

    def expected_observation_fraction(self):
        # class 0 is the observed/sensor class in this code.
        probs = torch.softmax(self.logits, dim=0)
        return probs[0].mean()

    def compute_logits_loss(
        self,
        logits_init,
        batch,
        budget_obs=0.001,
        tau=1.0,
        weight=1.0,
        hard=True,
    ):
        tgt_inp = batch.tgt.clone()
        logits = logits_init.to(self.device)

        # Same hard-forward / straight-through Gumbel-Softmax behavior as baseline.
        gs_output = F.gumbel_softmax(
            logits,
            tau=tau,
            hard=hard,
            dim=0,
        )[0, :, :, :]

        mask_input = gs_output.view(
            self.time_window,
            self.lat,
            self.lon,
        ).unsqueeze(0)

        normalized_selected_points = mask_input.mean()

        selected_tgt_inp = tgt_inp * mask_input
        selected_tgt_inp[mask_input == 0] = float("nan")

        L1, L2, theta = self.covariance_params()

        output = optimal_interpolation_chunk(
            selected_tgt_inp[0, 0],
            L1=L1,
            L2=L2,
            theta=theta,
            chunk_size=OI_CHUNK_SIZE,
        )
        output = output.unsqueeze(0).unsqueeze(0).to(tgt_inp.dtype)

        loss_mse = F.mse_loss(output, batch.tgt)

        # Same upper-budget penalty logic as the baseline. During warm-start
        # fine-tuning we activate the final penalty immediately instead of
        # repeating the from-scratch warmup that would encourage sensor growth.
        effective_weight = weight if normalized_selected_points >= budget_obs else 0.0
        loss = loss_mse + effective_weight * normalized_selected_points

        return (
            output,
            loss,
            mask_input,
            normalized_selected_points,
            loss_mse,
        )

    def forward(self, batch, budget_obs, tau, weight, n_draw):
        total_loss = 0
        total_points = 0
        total_mse = 0

        for _ in range(n_draw):
            (
                output,
                loss_draw,
                mask,
                normalized_selected_points,
                loss_mse,
            ) = self.compute_logits_loss(
                self.logits,
                batch,
                budget_obs=budget_obs,
                tau=tau,
                weight=weight,
                hard=True,
            )

            total_loss += loss_draw
            total_points += normalized_selected_points
            total_mse += loss_mse

            del output, mask

        mean_loss = total_loss / n_draw
        mean_points = total_points / n_draw
        mean_mse = total_mse / n_draw

        L1, L2, theta = self.covariance_params()
        return mean_loss, mean_points, mean_mse, L1, L2, theta


# =============================================================================
# Fixed-seed checkpoint evaluation
# =============================================================================

def hard_mask_from_fixed_seed(logits, seed):
    """
    Deterministic Gumbel-max draw. For a hard draw, tau does not change argmax.
    Returns the class-0 (observation) mask.
    """
    gen = torch.Generator(device=logits.device)
    gen.manual_seed(int(seed))

    u = torch.rand(
        logits.shape,
        dtype=logits.dtype,
        device=logits.device,
        generator=gen,
    )
    u = u.clamp_(1e-7, 1.0 - 1e-7)
    gumbel = -torch.log(-torch.log(u))

    cls = torch.argmax(logits + gumbel, dim=0)
    return (cls == 0).to(logits.dtype)


@torch.no_grad()
def evaluate_fixed(model, batch):
    """
    Mean normalized reconstruction MSE on fixed hard Gumbel draws.

    The same seeds are used for the baseline checkpoint and every later A2
    checkpoint, so checkpoint selection is paired and deterministic.
    """
    L1, L2, theta = model.covariance_params()

    mse_values = []
    obs_values = []

    for j in range(EVAL_DRAWS):
        mask = hard_mask_from_fixed_seed(model.logits, EVAL_SEED + j)
        mask_input = mask.view(
            model.time_window,
            model.lat,
            model.lon,
        ).unsqueeze(0)

        selected = batch.tgt * mask_input
        selected[mask_input == 0] = float("nan")

        output = optimal_interpolation_chunk(
            selected[0, 0],
            L1=L1,
            L2=L2,
            theta=theta,
            chunk_size=OI_CHUNK_SIZE,
        )
        output = output.unsqueeze(0).unsqueeze(0).to(batch.tgt.dtype)

        mse_values.append(float(F.mse_loss(output, batch.tgt).cpu()))
        obs_values.append(float(mask_input.mean().cpu()))

    expected_obs = float(model.expected_observation_fraction().cpu())

    return {
        "mse": float(np.mean(mse_values)),
        "mse_std": float(np.std(mse_values)),
        "sampled_obs": float(np.mean(obs_values)),
        "expected_obs": expected_obs,
    }


def cpu_state_dict(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


# =============================================================================
# Saving helpers
# =============================================================================

def canonicalize_lengths_and_angle(L1, L2, theta):
    """
    Reporting-only canonicalization:
      L_major >= L_minor
      theta_major in [0, pi)

    Training parameters themselves remain unconstrained.
    """
    L1 = float(L1)
    L2 = float(L2)
    theta = float(theta)

    if L1 >= L2:
        major, minor = L1, L2
        angle = theta
    else:
        major, minor = L2, L1
        angle = theta + np.pi / 2.0

    angle = angle % np.pi
    return major, minor, angle


def save_cumulative(
    records,
    logits_records,
    obs_prob_records,
):
    """
    Rewrite the cumulative files after every completed day.
    If the job is killed during the next day, all previous days remain saved.
    """
    if not records:
        return

    times = [r["time"] for r in records]

    results_ds = xr.Dataset(
        {
            "mean_loss": xr.DataArray(
                np.stack([r["mean_loss"] for r in records]),
                dims=["time", "iteration"],
                coords={"time": times, "iteration": np.arange(TOTAL_ITERS)},
            ),
            "mean_points": xr.DataArray(
                np.stack([r["mean_points"] for r in records]),
                dims=["time", "iteration"],
                coords={"time": times, "iteration": np.arange(TOTAL_ITERS)},
            ),
            "mean_mse": xr.DataArray(
                np.stack([r["mean_mse"] for r in records]),
                dims=["time", "iteration"],
                coords={"time": times, "iteration": np.arange(TOTAL_ITERS)},
            ),
            "L1": xr.DataArray(
                np.stack([r["L1"] for r in records]),
                dims=["time", "iteration"],
                coords={"time": times, "iteration": np.arange(TOTAL_ITERS)},
            ),
            "L2": xr.DataArray(
                np.stack([r["L2"] for r in records]),
                dims=["time", "iteration"],
                coords={"time": times, "iteration": np.arange(TOTAL_ITERS)},
            ),
            "theta": xr.DataArray(
                np.stack([r["theta"] for r in records]),
                dims=["time", "iteration"],
                coords={"time": times, "iteration": np.arange(TOTAL_ITERS)},
            ),
            "baseline_L": xr.DataArray(
                [r["baseline_L"] for r in records],
                dims=["time"],
                coords={"time": times},
            ),
            "baseline_saved_training_mse": xr.DataArray(
                [r["baseline_saved_training_mse"] for r in records],
                dims=["time"],
                coords={"time": times},
            ),
            "baseline_eval_mse": xr.DataArray(
                [r["baseline_eval_mse"] for r in records],
                dims=["time"],
                coords={"time": times},
            ),
            "baseline_eval_obs": xr.DataArray(
                [r["baseline_eval_obs"] for r in records],
                dims=["time"],
                coords={"time": times},
            ),
            "best_eval_mse": xr.DataArray(
                [r["best_eval_mse"] for r in records],
                dims=["time"],
                coords={"time": times},
            ),
            "best_eval_obs": xr.DataArray(
                [r["best_eval_obs"] for r in records],
                dims=["time"],
                coords={"time": times},
            ),
            "best_expected_obs": xr.DataArray(
                [r["best_expected_obs"] for r in records],
                dims=["time"],
                coords={"time": times},
            ),
            "best_iteration": xr.DataArray(
                [r["best_iteration"] for r in records],
                dims=["time"],
                coords={"time": times},
            ),
            "best_stage": xr.DataArray(
                np.asarray([r["best_stage"] for r in records], dtype="U24"),
                dims=["time"],
                coords={"time": times},
            ),
            "best_L1": xr.DataArray(
                [r["best_L1"] for r in records],
                dims=["time"],
                coords={"time": times},
            ),
            "best_L2": xr.DataArray(
                [r["best_L2"] for r in records],
                dims=["time"],
                coords={"time": times},
            ),
            "best_theta": xr.DataArray(
                [r["best_theta"] for r in records],
                dims=["time"],
                coords={"time": times},
            ),
            "best_L_major": xr.DataArray(
                [r["best_L_major"] for r in records],
                dims=["time"],
                coords={"time": times},
            ),
            "best_L_minor": xr.DataArray(
                [r["best_L_minor"] for r in records],
                dims=["time"],
                coords={"time": times},
            ),
            "best_theta_major": xr.DataArray(
                [r["best_theta_major"] for r in records],
                dims=["time"],
                coords={"time": times},
            ),
            "improvement_pct": xr.DataArray(
                [r["improvement_pct"] for r in records],
                dims=["time"],
                coords={"time": times},
            ),
        }
    )

    results_ds["L1"].attrs["units"] = "grid cells"
    results_ds["L2"].attrs["units"] = "grid cells"
    results_ds["theta"].attrs["units"] = "radians"
    results_ds["best_L1"].attrs["units"] = "grid cells"
    results_ds["best_L2"].attrs["units"] = "grid cells"
    results_ds["best_theta"].attrs["units"] = "radians"
    results_ds["best_L_major"].attrs["units"] = "grid cells"
    results_ds["best_L_minor"].attrs["units"] = "grid cells"
    results_ds["best_theta_major"].attrs["units"] = "radians"

    results_ds.attrs.update(
        {
            "covariance": "rotated anisotropic RBF (A2), warm-started from isotropic baseline",
            "stage1_iters": STAGE1_ITERS,
            "stage2_iters": STAGE2_ITERS,
            "n_draw": N_DRAW,
            "accumulation_steps": ACCUMULATION_STEPS,
            "oi_chunk_size": OI_CHUNK_SIZE,
            "budget_obs": BUDGET_OBS,
            "cov_lr": COV_LR,
            "logits_lr": LOGITS_LR,
            "eval_every": EVAL_EVERY,
            "eval_draws": EVAL_DRAWS,
            "eval_seed": EVAL_SEED,
            "checkpoint_rule": (
                "Keep baseline unless fixed-seed eval MSE improves while "
                "expected observation fraction remains within the nominal budget."
            ),
        }
    )

    results_path = OUTPUT_DIR / "results_natl60_A2_warmstart.nc"
    logits_path = OUTPUT_DIR / "logits_natl60_A2_warmstart.nc"
    obs_prob_path = OUTPUT_DIR / "obs_prob_natl60_A2_warmstart.nc"

    results_ds.to_netcdf(results_path)

    logits_all = xr.concat(
        logits_records,
        dim=xr.DataArray(times, dims="time", name="time"),
    )
    logits_all.attrs["representation"] = (
        "effective class-1 minus class-0 logit; "
        "observation probability = sigmoid(-logit)"
    )
    logits_all.to_netcdf(logits_path)

    obs_all = xr.concat(
        obs_prob_records,
        dim=xr.DataArray(times, dims="time", name="time"),
    )
    obs_all.to_netcdf(obs_prob_path)


# =============================================================================
# Load data + validated isotropic baseline
# =============================================================================

print(f"Device: {device}")
print(f"Opening data: {DATA_PATH}")
print(f"Baseline results: {BASELINE_RESULTS_PATH}")
print(f"Baseline logits:  {BASELINE_LOGITS_PATH}")

for required in [DATA_PATH, BASELINE_RESULTS_PATH, BASELINE_LOGITS_PATH]:
    if not Path(required).exists():
        raise FileNotFoundError(required)

tgt_ds = xr.open_dataset(DATA_PATH)
tgt_ds = tgt_ds.isel(time=slice(0, None, 15))

if tgt_ds.sizes["time"] != 25:
    raise RuntimeError(
        f"Expected the same 25 selected days as baseline, got {tgt_ds.sizes['time']}."
    )

baseline_results = xr.open_dataset(BASELINE_RESULTS_PATH)
baseline_logits_ds = xr.open_dataset(BASELINE_LOGITS_PATH)

if baseline_results.sizes.get("time") != 25:
    raise RuntimeError(
        f"Baseline results should have 25 days, got {baseline_results.sizes.get('time')}."
    )

if baseline_logits_ds.sizes.get("time") != 25:
    raise RuntimeError(
        f"Baseline logits should have 25 days, got {baseline_logits_ds.sizes.get('time')}."
    )

if "length_scale" not in baseline_results:
    raise KeyError("Baseline results do not contain 'length_scale'.")

if "mean_mse" not in baseline_results:
    raise KeyError("Baseline results do not contain 'mean_mse'.")

if "logits" not in baseline_logits_ds:
    raise KeyError("Baseline logits file does not contain 'logits'.")

START_DAY_CLAMPED = max(0, START_DAY)
END_DAY_CLAMPED = min(25, END_DAY)

if not (START_DAY_CLAMPED < END_DAY_CLAMPED):
    raise ValueError(
        f"Invalid day range [{START_DAY_CLAMPED}, {END_DAY_CLAMPED})."
    )

print(
    f"Warm-start configuration: days {START_DAY_CLAMPED + 1}..{END_DAY_CLAMPED}, "
    f"stage1={STAGE1_ITERS}, stage2={STAGE2_ITERS}, total={TOTAL_ITERS}"
)
print(
    f"LRs: covariance={COV_LR}, logits={LOGITS_LR} | "
    f"accumulation={ACCUMULATION_STEPS}"
)
print(
    f"Checkpoint eval: every {EVAL_EVERY} iters, {EVAL_DRAWS} fixed draws, "
    f"seed={EVAL_SEED}"
)


# =============================================================================
# Fine-tune each selected day independently
# =============================================================================

records = []
logits_records = []
obs_prob_records = []

TrainingItem = namedtuple("TrainingItem", ["input", "tgt"])

# Preserve the validated baseline's near-constant temperature behavior.
baseline_temperature = START_TEMP * (MIN_TEMP / START_TEMP) ** (1 / (800 - 1))

for day_index in range(START_DAY_CLAMPED, END_DAY_CLAMPED):
    time_step = tgt_ds.time.isel(time=day_index)

    print(
        f"\n=== Day {day_index + 1:02d}/25 | {str(time_step.values)[:10]} ===",
        flush=True,
    )

    inp_da = tgt_ds.ssh.sel(time=time_step)
    inp_da_GS = inp_da.sel(
        lat=slice(28, 45.0),
        lon=slice(-66, -49.0),
    )

    border_size = 30
    lat_shape = len(inp_da_GS.lat)
    lon_shape = len(inp_da_GS.lon)

    inp_da_GS_crop = inp_da_GS.isel(
        lat=slice(border_size, lat_shape - border_size),
        lon=slice(border_size, lon_shape - border_size),
    ).fillna(0.0)

    if len(inp_da_GS_crop.shape) < 3:
        time = 1
        lat, lon = inp_da_GS_crop.shape
    else:
        time, lat, lon = inp_da_GS_crop.shape

    if (lat, lon) != (280, 280):
        raise RuntimeError(f"Expected final 280x280 crop, got {(lat, lon)}.")

    mean_tgt = inp_da_GS_crop.mean().item()
    std_tgt = inp_da_GS_crop.std().item()

    tens_inp_da = torch.from_numpy(
        inp_da_GS_crop.values
    ).float().to(device)

    tens_inp_da = (tens_inp_da - mean_tgt) / std_tgt
    tens_inp_da = tens_inp_da.unsqueeze(0).unsqueeze(0)
    tens_inp_da.requires_grad_(False)

    batch = TrainingItem(
        input=tens_inp_da,
        tgt=tens_inp_da,
    )

    baseline_L = float(
        baseline_results["length_scale"]
        .isel(time=day_index, iteration=-1)
        .values
    )
    baseline_saved_training_mse = float(
        baseline_results["mean_mse"]
        .isel(time=day_index, iteration=-1)
        .values
    )
    baseline_effective_logit = (
        baseline_logits_ds["logits"]
        .isel(time=day_index)
        .values
        .astype(np.float32)
    )

    model = GSModel(time, lat, lon, device).to(device)
    model.initialize_from_baseline(
        baseline_effective_logit=baseline_effective_logit,
        baseline_L=baseline_L,
    )

    # -------------------------------------------------------------------------
    # Baseline checkpoint: A2 is exactly isotropic here.
    # -------------------------------------------------------------------------
    baseline_eval = evaluate_fixed(model, batch)
    baseline_expected_obs = baseline_eval["expected_obs"]
    budget_limit = max(BUDGET_OBS, baseline_expected_obs)

    best_state = cpu_state_dict(model)
    best_eval = dict(baseline_eval)
    best_stage = "baseline"
    best_iteration = 0

    with torch.no_grad():
        L1_0, L2_0, th_0 = model.covariance_params()

    print(
        f"baseline | saved_train_mse={baseline_saved_training_mse:.6f} | "
        f"fixed_eval_mse={baseline_eval['mse']:.6f} ± {baseline_eval['mse_std']:.6f} | "
        f"expected_obs={baseline_expected_obs:.6f} | "
        f"L={baseline_L:.3f}",
        flush=True,
    )

    mean_loss_list = []
    mean_points_list = []
    mean_mse_list = []
    L1_list = []
    L2_list = []
    theta_list = []

    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    def run_training_stage(
        stage_name,
        n_iters,
        optimizer,
        train_logits,
        global_offset,
        starting_best_state,
        starting_best_eval,
        starting_best_stage,
        starting_best_iteration,
    ):
        stage_best_state = starting_best_state
        stage_best_eval = dict(starting_best_eval)
        stage_best_stage = starting_best_stage
        stage_best_iteration = starting_best_iteration

        model.logits.requires_grad_(train_logits)

        for local_k in range(n_iters):
            global_k = global_offset + local_k
            optimizer.zero_grad(set_to_none=True)

            total_loss_report = 0.0
            total_points_report = 0.0
            total_mse_report = 0.0

            # No from-scratch warmup here. We are already starting from a trained
            # baseline, so the final budget constraint stays active.
            weight = MAX_WEIGHT

            for _ in range(ACCUMULATION_STEPS):
                with autocast(
                    device_type="cuda",
                    enabled=(device.type == "cuda"),
                ):
                    (
                        mean_loss,
                        normalized_selected_points,
                        loss_mse,
                        L1,
                        L2,
                        theta,
                    ) = model(
                        batch,
                        budget_obs=BUDGET_OBS,
                        tau=baseline_temperature,
                        n_draw=N_DRAW,
                        weight=weight,
                    )

                    scaled_loss = mean_loss / ACCUMULATION_STEPS

                total_loss_report += (
                    mean_loss.detach() / ACCUMULATION_STEPS
                )
                total_points_report += (
                    normalized_selected_points.detach() / ACCUMULATION_STEPS
                )
                total_mse_report += (
                    loss_mse.detach() / ACCUMULATION_STEPS
                )

                scaler.scale(scaled_loss).backward()

            # At exact isotropy theta is unidentifiable. Keep it fixed only for
            # the first covariance update; L1/L2 can then separate and theta can
            # become identifiable naturally.
            if stage_name == "covariance_only" and local_k == 0:
                model.theta.grad = None

            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                L1_now, L2_now, theta_now = model.covariance_params()

            mean_loss_list.append(float(total_loss_report.cpu()))
            mean_points_list.append(float(total_points_report.cpu()))
            mean_mse_list.append(float(total_mse_report.cpu()))
            L1_list.append(float(L1_now.cpu()))
            L2_list.append(float(L2_now.cpu()))
            theta_list.append(float(theta_now.cpu()))

            should_eval = (
                (local_k + 1) % EVAL_EVERY == 0
                or (local_k + 1) == n_iters
            )

            if should_eval:
                current_eval = evaluate_fixed(model, batch)

                # Nominal paper budget is an upper bound. The reconstructed
                # baseline is always allowed if it is microscopically above it.
                budget_ok = (
                    current_eval["expected_obs"] <= budget_limit + 1e-12
                )

                rel_improvement = (
                    (stage_best_eval["mse"] - current_eval["mse"])
                    / stage_best_eval["mse"]
                    if stage_best_eval["mse"] > 0
                    else 0.0
                )

                improved = (
                    budget_ok
                    and current_eval["mse"] < stage_best_eval["mse"]
                    and rel_improvement >= MIN_REL_IMPROVEMENT
                )

                status = ""
                if improved:
                    status = "  <-- NEW BEST"
                elif not budget_ok:
                    status = "  [over expected budget]"

                ratio = max(L1_list[-1], L2_list[-1]) / min(
                    L1_list[-1], L2_list[-1]
                )

                print(
                    f"{stage_name:15s} iter {local_k + 1:03d}/{n_iters} "
                    f"(global {global_k + 1:03d}/{TOTAL_ITERS}) | "
                    f"train_mse={mean_mse_list[-1]:.6f} | "
                    f"eval_mse={current_eval['mse']:.6f} | "
                    f"exp_obs={current_eval['expected_obs']:.6f} | "
                    f"L1={L1_list[-1]:.3f} L2={L2_list[-1]:.3f} "
                    f"ratio={ratio:.3f} theta={np.degrees(theta_list[-1]):.2f}°"
                    f"{status}",
                    flush=True,
                )

                if improved:
                    stage_best_state = cpu_state_dict(model)
                    stage_best_eval = dict(current_eval)
                    stage_best_stage = stage_name
                    stage_best_iteration = global_k + 1

        return {
            "best_state": stage_best_state,
            "best_eval": stage_best_eval,
            "best_stage": stage_best_stage,
            "best_iteration": stage_best_iteration,
        }

    # -------------------------------------------------------------------------
    # Stage 1: same learned sensor placement, richer covariance only.
    # -------------------------------------------------------------------------
    model.logits.requires_grad_(False)

    optimizer_stage1 = optim.Adam(
        [model.raw_L1, model.raw_L2, model.theta],
        lr=COV_LR,
    )

    updates = run_training_stage(
        stage_name="covariance_only",
        n_iters=STAGE1_ITERS,
        optimizer=optimizer_stage1,
        train_logits=False,
        global_offset=0,
        starting_best_state=best_state,
        starting_best_eval=best_eval,
        starting_best_stage=best_stage,
        starting_best_iteration=best_iteration,
    )

    best_state = updates["best_state"]
    best_eval = updates["best_eval"]
    best_stage = updates["best_stage"]
    best_iteration = updates["best_iteration"]

    # Start joint fine-tuning from the best solution found so far, not from a
    # potentially worse last covariance-only iterate.
    model.load_state_dict(best_state)
    model.to(device)
    model.logits.requires_grad_(True)

    # -------------------------------------------------------------------------
    # Stage 2: jointly fine-tune covariance + sensor logits.
    # -------------------------------------------------------------------------
    optimizer_stage2 = optim.Adam(
        [
            {"params": [model.raw_L1, model.raw_L2, model.theta], "lr": COV_LR},
            {"params": [model.logits], "lr": LOGITS_LR},
        ]
    )

    updates = run_training_stage(
        stage_name="joint",
        n_iters=STAGE2_ITERS,
        optimizer=optimizer_stage2,
        train_logits=True,
        global_offset=STAGE1_ITERS,
        starting_best_state=best_state,
        starting_best_eval=best_eval,
        starting_best_stage=best_stage,
        starting_best_iteration=best_iteration,
    )

    best_state = updates["best_state"]
    best_eval = updates["best_eval"]
    best_stage = updates["best_stage"]
    best_iteration = updates["best_iteration"]

    # IMPORTANT: always restore the best checkpoint. If anisotropy/joint
    # fine-tuning never improved the fixed paired evaluation, this is literally
    # the initial isotropic baseline checkpoint.
    model.load_state_dict(best_state)
    model.to(device)

    final_eval = evaluate_fixed(model, batch)

    with torch.no_grad():
        best_L1, best_L2, best_theta = model.covariance_params()
        best_L1 = float(best_L1.cpu())
        best_L2 = float(best_L2.cpu())
        best_theta = float(best_theta.cpu())

        effective_logit = (
            model.effective_logit()[0].detach().cpu().numpy().astype(np.float32)
        )
        obs_prob = (
            torch.sigmoid(-model.effective_logit()[0])
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    best_L_major, best_L_minor, best_theta_major = canonicalize_lengths_and_angle(
        best_L1,
        best_L2,
        best_theta,
    )

    improvement_pct = (
        100.0 * (baseline_eval["mse"] - final_eval["mse"]) / baseline_eval["mse"]
        if baseline_eval["mse"] > 0
        else 0.0
    )

    print(
        f"BEST day {day_index + 1:02d}: stage={best_stage}, "
        f"iter={best_iteration}, "
        f"eval_mse {baseline_eval['mse']:.6f} -> {final_eval['mse']:.6f} "
        f"({improvement_pct:+.2f}%), "
        f"expected_obs={final_eval['expected_obs']:.6f}, "
        f"Lmajor/Lminor={best_L_major / best_L_minor:.3f}, "
        f"theta_major={np.degrees(best_theta_major):.2f}°",
        flush=True,
    )

    # Histories always have TOTAL_ITERS entries.
    record = {
        "time": time_step.values,
        "mean_loss": np.asarray(mean_loss_list, dtype=np.float64),
        "mean_points": np.asarray(mean_points_list, dtype=np.float64),
        "mean_mse": np.asarray(mean_mse_list, dtype=np.float64),
        "L1": np.asarray(L1_list, dtype=np.float32),
        "L2": np.asarray(L2_list, dtype=np.float32),
        "theta": np.asarray(theta_list, dtype=np.float32),
        "baseline_L": baseline_L,
        "baseline_saved_training_mse": baseline_saved_training_mse,
        "baseline_eval_mse": baseline_eval["mse"],
        "baseline_eval_obs": baseline_eval["expected_obs"],
        "best_eval_mse": final_eval["mse"],
        "best_eval_obs": final_eval["sampled_obs"],
        "best_expected_obs": final_eval["expected_obs"],
        "best_iteration": best_iteration,
        "best_stage": best_stage,
        "best_L1": best_L1,
        "best_L2": best_L2,
        "best_theta": best_theta,
        "best_L_major": best_L_major,
        "best_L_minor": best_L_minor,
        "best_theta_major": best_theta_major,
        "improvement_pct": improvement_pct,
    }
    records.append(record)

    logits_records.append(
        xr.DataArray(
            effective_logit,
            dims=["lat", "lon"],
            coords={
                "lat": inp_da_GS_crop.lat,
                "lon": inp_da_GS_crop.lon,
            },
            name="logits",
        )
    )

    obs_prob_records.append(
        xr.DataArray(
            obs_prob,
            dims=["lat", "lon"],
            coords={
                "lat": inp_da_GS_crop.lat,
                "lon": inp_da_GS_crop.lon,
            },
            name="obs_prob",
        )
    )

    # Save cumulative NetCDFs immediately after every completed day.
    save_cumulative(records, logits_records, obs_prob_records)

    # Also save an explicit per-day PyTorch checkpoint for maximum recoverability.
    checkpoint_path = OUTPUT_DIR / f"day_{day_index + 1:02d}_best.pt"
    torch.save(
        {
            "day_index": day_index,
            "time": str(time_step.values),
            "state_dict": cpu_state_dict(model),
            "baseline_L": baseline_L,
            "baseline_eval": baseline_eval,
            "best_eval": final_eval,
            "best_stage": best_stage,
            "best_iteration": best_iteration,
            "mean_tgt": mean_tgt,
            "std_tgt": std_tgt,
        },
        checkpoint_path,
    )

    print(
        f"Saved day {day_index + 1:02d} safely to {OUTPUT_DIR}",
        flush=True,
    )

    del model, tens_inp_da, batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


print("\nDone.")
print("Saved cumulative outputs after every completed day:")
print(f"  {OUTPUT_DIR / 'results_natl60_A2_warmstart.nc'}")
print(f"  {OUTPUT_DIR / 'logits_natl60_A2_warmstart.nc'}")
print(f"  {OUTPUT_DIR / 'obs_prob_natl60_A2_warmstart.nc'}")
print(f"  {OUTPUT_DIR / 'day_XX_best.pt'}")
