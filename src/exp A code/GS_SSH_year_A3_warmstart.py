import os
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
# A3: spatially varying covariance geometry
# =============================================================================
#
# A2 used one global rotated anisotropic RBF geometry per day:
#
#   phi_A2(x, y) = [ (cos(theta)x + sin(theta)y)/L1,
#                     (-sin(theta)x + cos(theta)y)/L2 ]
#
# A3 embeds this exactly in R^4 and adds a bounded learned spatial warp:
#
#   phi_A3(r) = [phi_A2(r), 0, 0] + g_eta(r_norm)
#
#   g_eta : R^2 -> R^4
#           2 -> 64 -> 64 -> 4
#           tanh   tanh   tanh(output)
#
# alpha is fixed to 1.0, as decided.  The final linear layer of g_eta is
# initialized to exactly zero, so the initial A3 model is EXACTLY A2.
#
# Covariance is then
#
#   C(r_i, r_j) = sigma_f^2 * exp(-0.5 * ||phi_A3(r_i)-phi_A3(r_j)||^2)
#
# which is a valid positive-semidefinite Gaussian/RBF kernel by construction.
# =============================================================================


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


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

# Warm-start from the FINISHED A2 experiment, not from the isotropic baseline.
A2_RESULTS_PATH = os.environ.get(
    "A2_RESULTS_PATH",
    first_existing(
        SCRIPT_DIR / "results_natl60_A2_warmstart.nc",
        SCRIPT_DIR / "outputs_A2_warmstart" / "results_natl60_A2_warmstart.nc",
    ),
)
A2_LOGITS_PATH = os.environ.get(
    "A2_LOGITS_PATH",
    first_existing(
        SCRIPT_DIR / "logits_natl60_A2_warmstart.nc",
        SCRIPT_DIR / "outputs_A2_warmstart" / "logits_natl60_A2_warmstart.nc",
    ),
)

OUTPUT_DIR = Path(
    os.environ.get(
        "GS_OUTPUT_DIR",
        str(SCRIPT_DIR / "outputs_A3_warmstart"),
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Stage 1 learns ONLY the new spatial warp.  Stage 2 jointly fine-tunes the
# warp, A2 global geometry, and sensor logits.
STAGE1_ITERS = int(os.environ.get("A3_STAGE1_ITERS", "200"))
STAGE2_ITERS = int(os.environ.get("A3_STAGE2_ITERS", "300"))
TOTAL_ITERS = STAGE1_ITERS + STAGE2_ITERS

# Same memory-safe stochastic gradient accumulation used by A2.
N_DRAW = 1
ACCUMULATION_STEPS = int(os.environ.get("A3_ACCUMULATION_STEPS", "60"))

# Fine-tuning LRs.  A2 is already optimized, so global covariance/logit LRs
# are kept smaller than in the A2 warm-start run.
WARP_LR = float(os.environ.get("A3_WARP_LR", "0.005"))
COV_LR = float(os.environ.get("A3_COV_LR", "0.005"))
LOGITS_LR = float(os.environ.get("A3_LOGITS_LR", "0.005"))

RATE = 0.999
BUDGET_OBS = 1.0 - RATE
MAX_WEIGHT = 75.0
START_TEMP = 15.0
MIN_TEMP = 4.0
OI_CHUNK_SIZE = int(os.environ.get("A3_OI_CHUNK_SIZE", "5000"))

# Architecture agreed for A3.
WARP_HIDDEN = 64
WARP_DIM = 4
ALPHA = 1.0

# New fixed masks for A3 checkpoint selection.  This deliberately differs
# from A2's 271828 checkpoint-selection seed.
EVAL_EVERY = int(os.environ.get("A3_EVAL_EVERY", "50"))
EVAL_DRAWS = int(os.environ.get("A3_EVAL_DRAWS", "32"))
EVAL_SEED = int(os.environ.get("A3_EVAL_SEED", "314159"))
MIN_REL_IMPROVEMENT = float(os.environ.get("A3_MIN_REL_IMPROVEMENT", "0.0"))

# Reproducible network initialization / training randomness per day.
MODEL_SEED = int(os.environ.get("A3_MODEL_SEED", "424242"))

# Optional day chunking / smoke tests.
START_DAY = int(os.environ.get("GS_START_DAY", "0"))
END_DAY = int(os.environ.get("GS_END_DAY", "25"))


# =============================================================================
# Helpers
# =============================================================================


def _as_tensor(value, *, device, dtype=torch.float32):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, dtype=dtype, device=device)


def inverse_softplus(y):
    y = torch.as_tensor(y, dtype=torch.float32)
    return y + torch.log(-torch.expm1(-y))


def embedded_rbf_covariance(z1, z2, sigma_f):
    """RBF covariance in the learned latent geometry."""
    delta = z1.unsqueeze(1) - z2.unsqueeze(0)
    sqdist = (delta * delta).sum(dim=-1)
    return sigma_f**2 * torch.exp(-0.5 * sqdist)


# =============================================================================
# A3 model
# =============================================================================


class SpatialWarp(nn.Module):
    """
    g_eta: R^2 -> R^4

    normalized (x,y)
      -> Linear(2,64) -> tanh
      -> Linear(64,64) -> tanh
      -> Linear(64,4) -> tanh

    The final linear layer is exactly zero-initialized, so g_eta == 0 at
    initialization regardless of the random hidden layers.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, WARP_HIDDEN)
        self.fc2 = nn.Linear(WARP_HIDDEN, WARP_HIDDEN)
        self.fc3 = nn.Linear(WARP_HIDDEN, WARP_DIM)

        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, xy_normalized):
        h = torch.tanh(self.fc1(xy_normalized))
        h = torch.tanh(self.fc2(h))
        return torch.tanh(self.fc3(h))


class GSModelA3(nn.Module):
    def __init__(self, time, lat, lon, device):
        super().__init__()

        self.logits = nn.Parameter(torch.zeros((2, time, lat, lon)))
        self.time_window = time
        self.lat = lat
        self.lon = lon
        self.device = device

        # A2 global geometry, initialized from the saved best A2 solution.
        self.raw_L1 = nn.Parameter(torch.tensor(25.0, dtype=torch.float32))
        self.raw_L2 = nn.Parameter(torch.tensor(25.0, dtype=torch.float32))
        self.theta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # New A3 component.
        self.warp = SpatialWarp()

        # Fixed coordinate grid.  Coordinates follow the original code's
        # [x,y] = [lon-index, lat-index] convention.
        yy, xx = torch.meshgrid(
            torch.arange(lat, dtype=torch.float32),
            torch.arange(lon, dtype=torch.float32),
            indexing="ij",
        )
        xy = torch.stack([xx, yy], dim=-1)

        x_norm = 2.0 * xx / max(lon - 1, 1) - 1.0
        y_norm = 2.0 * yy / max(lat - 1, 1) - 1.0
        xy_norm = torch.stack([x_norm, y_norm], dim=-1)

        self.register_buffer("grid_xy", xy, persistent=False)
        self.register_buffer("grid_xy_norm", xy_norm, persistent=False)

    def covariance_params(self):
        eps = 1e-6
        L1 = F.softplus(self.raw_L1) + eps
        L2 = F.softplus(self.raw_L2) + eps
        return L1, L2, self.theta

    @torch.no_grad()
    def initialize_from_a2(
        self,
        a2_effective_logit,
        a2_L1,
        a2_L2,
        a2_theta,
    ):
        """Initialize A3 to be exactly the saved A2 solution."""
        self.logits.zero_()

        t = torch.as_tensor(
            a2_effective_logit,
            dtype=self.logits.dtype,
            device=self.logits.device,
        )
        if t.ndim == 2:
            t = t.unsqueeze(0)

        if tuple(t.shape) != tuple(self.logits[1].shape):
            raise ValueError(
                f"A2 logits shape {tuple(t.shape)} does not match "
                f"model shape {tuple(self.logits[1].shape)}."
            )

        self.logits[1].copy_(t)

        target_L1 = torch.tensor(
            float(a2_L1) - 1e-6,
            device=self.raw_L1.device,
            dtype=self.raw_L1.dtype,
        ).clamp_min(1e-6)
        target_L2 = torch.tensor(
            float(a2_L2) - 1e-6,
            device=self.raw_L2.device,
            dtype=self.raw_L2.dtype,
        ).clamp_min(1e-6)

        self.raw_L1.copy_(inverse_softplus(target_L1).to(self.raw_L1.device))
        self.raw_L2.copy_(inverse_softplus(target_L2).to(self.raw_L2.device))
        self.theta.copy_(
            torch.tensor(
                float(a2_theta),
                dtype=self.theta.dtype,
                device=self.theta.device,
            )
        )

        # Re-enforce exact A2 initialization even if this model object were
        # reused in an interactive notebook.
        nn.init.zeros_(self.warp.fc3.weight)
        nn.init.zeros_(self.warp.fc3.bias)

    def effective_logit(self):
        return self.logits[1] - self.logits[0]

    def expected_observation_fraction(self):
        probs = torch.softmax(self.logits, dim=0)
        return probs[0].mean()

    def geometry_embedding(self):
        """
        Return phi_A3 for every grid cell, shape [lat, lon, 4].

        First two dimensions contain the exact A2 global geometry.  Last two
        start at zero.  The learned g_eta correction is then added with
        fixed alpha = 1.
        """
        L1, L2, theta = self.covariance_params()

        x = self.grid_xy[..., 0]
        y = self.grid_xy[..., 1]

        c = torch.cos(theta)
        s = torch.sin(theta)

        z1 = (c * x + s * y) / L1
        z2 = (-s * x + c * y) / L2
        zeros = torch.zeros_like(z1)

        base4 = torch.stack([z1, z2, zeros, zeros], dim=-1)

        flat_norm = self.grid_xy_norm.reshape(-1, 2)
        g = self.warp(flat_norm).reshape(self.lat, self.lon, WARP_DIM)

        return base4 + ALPHA * g

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

        phi_grid = self.geometry_embedding()

        output = optimal_interpolation_chunk_a3(
            selected_tgt_inp[0, 0],
            phi_grid=phi_grid,
            chunk_size=OI_CHUNK_SIZE,
        )
        output = output.unsqueeze(0).unsqueeze(0).to(tgt_inp.dtype)

        loss_mse = F.mse_loss(output, batch.tgt)

        effective_weight = weight if normalized_selected_points >= budget_obs else 0.0
        loss = loss_mse + effective_weight * normalized_selected_points

        return output, loss, mask_input, normalized_selected_points, loss_mse

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
# Memory-safe optimal interpolation in learned A3 geometry
# =============================================================================


def optimal_interpolation_chunk_a3(
    data_with_nans,
    phi_grid,
    sigma_f=1.0,
    sigma_n=0.1,
    chunk_size=5000,
):
    device_ = data_with_nans.device

    observed_indices = ~torch.isnan(data_with_nans)
    missing_indices = torch.isnan(data_with_nans)

    observed_values = data_with_nans[observed_indices].float().to(device_)

    # The expensive NN embedding was computed once for the whole 280x280 grid.
    # We now only select observed/missing latent positions.
    observed_phi = phi_grid[observed_indices]
    missing_phi = phi_grid[missing_indices]

    sigma_f = _as_tensor(sigma_f, device=device_)
    sigma_n = _as_tensor(sigma_n, device=device_)

    K_oo = embedded_rbf_covariance(observed_phi, observed_phi, sigma_f)
    eye = torch.eye(K_oo.size(0), device=device_, dtype=K_oo.dtype)
    K_oo = K_oo + sigma_n**2 * eye

    L_chol = torch.linalg.cholesky(K_oo + 1e-6 * eye)
    alpha_oi = torch.cholesky_solve(observed_values.unsqueeze(1), L_chol)

    num_missing = missing_phi.size(0)
    pred_list = []

    for start in range(0, num_missing, chunk_size):
        end = min(start + chunk_size, num_missing)
        K_s_chunk = embedded_rbf_covariance(
            observed_phi,
            missing_phi[start:end],
            sigma_f,
        )
        pred_chunk = K_s_chunk.t().matmul(alpha_oi).squeeze(1)
        pred_list.append(pred_chunk)
        del K_s_chunk, pred_chunk

    if pred_list:
        mean_pred = torch.cat(pred_list, dim=0)
    else:
        mean_pred = torch.empty(0, device=device_, dtype=torch.float32)

    reconstructed_data = data_with_nans.clone()
    if mean_pred.numel() > 0:
        reconstructed_data[missing_indices] = mean_pred.to(reconstructed_data.dtype)

    return reconstructed_data


# =============================================================================
# Fixed-seed checkpoint evaluation
# =============================================================================


def hard_mask_from_fixed_seed(logits, seed):
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
    mse_values = []
    obs_values = []

    # Geometry does not depend on the mask, so evaluate it once per checkpoint.
    phi_grid = model.geometry_embedding()

    for j in range(EVAL_DRAWS):
        mask = hard_mask_from_fixed_seed(model.logits, EVAL_SEED + j)
        mask_input = mask.view(
            model.time_window,
            model.lat,
            model.lon,
        ).unsqueeze(0)

        selected = batch.tgt * mask_input
        selected[mask_input == 0] = float("nan")

        output = optimal_interpolation_chunk_a3(
            selected[0, 0],
            phi_grid=phi_grid,
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
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# =============================================================================
# Saving helpers
# =============================================================================


def canonicalize_lengths_and_angle(L1, L2, theta):
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


def atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_cumulative(
    records,
    logits_records,
    obs_prob_records,
    warp_records,
    model_bundle,
):
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
            "a2_L1": xr.DataArray(
                [r["a2_L1"] for r in records], dims=["time"], coords={"time": times}
            ),
            "a2_L2": xr.DataArray(
                [r["a2_L2"] for r in records], dims=["time"], coords={"time": times}
            ),
            "a2_theta": xr.DataArray(
                [r["a2_theta"] for r in records], dims=["time"], coords={"time": times}
            ),
            "a2_eval_mse": xr.DataArray(
                [r["a2_eval_mse"] for r in records], dims=["time"], coords={"time": times}
            ),
            "a2_eval_obs": xr.DataArray(
                [r["a2_eval_obs"] for r in records], dims=["time"], coords={"time": times}
            ),
            "best_eval_mse": xr.DataArray(
                [r["best_eval_mse"] for r in records], dims=["time"], coords={"time": times}
            ),
            "best_eval_obs": xr.DataArray(
                [r["best_eval_obs"] for r in records], dims=["time"], coords={"time": times}
            ),
            "best_expected_obs": xr.DataArray(
                [r["best_expected_obs"] for r in records], dims=["time"], coords={"time": times}
            ),
            "best_iteration": xr.DataArray(
                [r["best_iteration"] for r in records], dims=["time"], coords={"time": times}
            ),
            "best_stage": xr.DataArray(
                np.asarray([r["best_stage"] for r in records], dtype="U24"),
                dims=["time"],
                coords={"time": times},
            ),
            "best_L1": xr.DataArray(
                [r["best_L1"] for r in records], dims=["time"], coords={"time": times}
            ),
            "best_L2": xr.DataArray(
                [r["best_L2"] for r in records], dims=["time"], coords={"time": times}
            ),
            "best_theta": xr.DataArray(
                [r["best_theta"] for r in records], dims=["time"], coords={"time": times}
            ),
            "best_L_major": xr.DataArray(
                [r["best_L_major"] for r in records], dims=["time"], coords={"time": times}
            ),
            "best_L_minor": xr.DataArray(
                [r["best_L_minor"] for r in records], dims=["time"], coords={"time": times}
            ),
            "best_theta_major": xr.DataArray(
                [r["best_theta_major"] for r in records], dims=["time"], coords={"time": times}
            ),
            "improvement_pct_vs_A2": xr.DataArray(
                [r["improvement_pct_vs_A2"] for r in records],
                dims=["time"],
                coords={"time": times},
            ),
        }
    )

    for name in ["L1", "L2", "a2_L1", "a2_L2", "best_L1", "best_L2", "best_L_major", "best_L_minor"]:
        results_ds[name].attrs["units"] = "grid cells"
    for name in ["theta", "a2_theta", "best_theta", "best_theta_major"]:
        results_ds[name].attrs["units"] = "radians"

    results_ds.attrs.update(
        {
            "experiment": "A3 spatially varying covariance geometry warm-started from A2",
            "covariance": "RBF in learned 4D coordinate embedding",
            "embedding": "[phi_A2_1, phi_A2_2, 0, 0] + g_eta(normalized_xy)",
            "warp_architecture": "2-64-64-4 with tanh hidden/output activations",
            "warp_alpha": ALPHA,
            "warp_final_layer_zero_initialized": 1,
            "stage1": "warp only; A2 geometry and sensor logits frozen",
            "stage2": "joint warp + A2 geometry + sensor logits",
            "stage1_iters": STAGE1_ITERS,
            "stage2_iters": STAGE2_ITERS,
            "n_draw": N_DRAW,
            "accumulation_steps": ACCUMULATION_STEPS,
            "oi_chunk_size": OI_CHUNK_SIZE,
            "budget_obs": BUDGET_OBS,
            "warp_lr": WARP_LR,
            "cov_lr": COV_LR,
            "logits_lr": LOGITS_LR,
            "eval_every": EVAL_EVERY,
            "eval_draws": EVAL_DRAWS,
            "eval_seed": EVAL_SEED,
            "checkpoint_rule": (
                "Keep A2 unless fixed-seed A3 eval MSE improves while expected "
                "observation fraction remains within the allowed A2/paper budget."
            ),
        }
    )

    results_path = OUTPUT_DIR / "results_natl60_A3_warmstart.nc"
    logits_path = OUTPUT_DIR / "logits_natl60_A3_warmstart.nc"
    obs_prob_path = OUTPUT_DIR / "obs_prob_natl60_A3_warmstart.nc"
    warp_path = OUTPUT_DIR / "warp_natl60_A3_warmstart.nc"
    models_path = OUTPUT_DIR / "models_natl60_A3_warmstart.pt"

    results_ds.to_netcdf(results_path)

    logits_all = xr.concat(
        logits_records,
        dim=xr.DataArray(times, dims="time", name="time"),
    )
    logits_all.attrs["representation"] = (
        "effective class-1 minus class-0 logit; observation probability = sigmoid(-logit)"
    )
    logits_all.to_netcdf(logits_path)

    obs_all = xr.concat(
        obs_prob_records,
        dim=xr.DataArray(times, dims="time", name="time"),
    )
    obs_all.to_netcdf(obs_prob_path)

    warp_all = xr.concat(
        warp_records,
        dim=xr.DataArray(times, dims="time", name="time"),
    )
    warp_all.attrs["description"] = (
        "Final learned g_eta(x,y) field. A3 embedding = embedded A2 geometry + this field."
    )
    warp_all.attrs["alpha"] = ALPHA
    warp_all.to_netcdf(warp_path)

    atomic_torch_save(
        {
            "architecture": "2-64-64-4 tanh",
            "alpha": ALPHA,
            "warp_dim": WARP_DIM,
            "days": model_bundle,
        },
        models_path,
    )


# =============================================================================
# Load data + completed A2 outputs
# =============================================================================

print(f"Device: {device}")
print(f"Opening data: {DATA_PATH}")
print(f"A2 results: {A2_RESULTS_PATH}")
print(f"A2 logits:  {A2_LOGITS_PATH}")

for required in [DATA_PATH, A2_RESULTS_PATH, A2_LOGITS_PATH]:
    if not Path(required).exists():
        raise FileNotFoundError(required)

tgt_ds = xr.open_dataset(DATA_PATH)
tgt_ds = tgt_ds.isel(time=slice(0, None, 15))

if tgt_ds.sizes["time"] != 25:
    raise RuntimeError(
        f"Expected the same 25 selected days as A2, got {tgt_ds.sizes['time']}."
    )

a2_results = xr.open_dataset(A2_RESULTS_PATH)
a2_logits_ds = xr.open_dataset(A2_LOGITS_PATH)

if a2_results.sizes.get("time") != 25:
    raise RuntimeError(
        f"A2 results should have 25 days, got {a2_results.sizes.get('time')}."
    )
if a2_logits_ds.sizes.get("time") != 25:
    raise RuntimeError(
        f"A2 logits should have 25 days, got {a2_logits_ds.sizes.get('time')}."
    )

for name in ["best_L1", "best_L2", "best_theta"]:
    if name not in a2_results:
        raise KeyError(f"A2 results do not contain '{name}'.")
if "logits" not in a2_logits_ds:
    raise KeyError("A2 logits file does not contain 'logits'.")

START_DAY_CLAMPED = max(0, START_DAY)
END_DAY_CLAMPED = min(25, END_DAY)
if not (START_DAY_CLAMPED < END_DAY_CLAMPED):
    raise ValueError(f"Invalid day range [{START_DAY_CLAMPED}, {END_DAY_CLAMPED}).")

print(
    f"A3 configuration: days {START_DAY_CLAMPED + 1}..{END_DAY_CLAMPED}, "
    f"stage1={STAGE1_ITERS}, stage2={STAGE2_ITERS}, total={TOTAL_ITERS}"
)
print("Warp: 2 -> 64 -> 64 -> 4, tanh, alpha=1, zero-init final layer")
print(
    f"LRs: warp={WARP_LR}, covariance={COV_LR}, logits={LOGITS_LR} | "
    f"accumulation={ACCUMULATION_STEPS}"
)
print(
    f"Checkpoint eval: every {EVAL_EVERY} iters, {EVAL_DRAWS} fixed draws, "
    f"seed={EVAL_SEED}"
)


# =============================================================================
# Train each of the 25 selected days independently
# =============================================================================

records = []
logits_records = []
obs_prob_records = []
warp_records = []
model_bundle = {}

TrainingItem = namedtuple("TrainingItem", ["input", "tgt"])

# Preserve the validated baseline/A2 near-constant temperature behavior.
baseline_temperature = START_TEMP * (MIN_TEMP / START_TEMP) ** (1 / (800 - 1))

for day_index in range(START_DAY_CLAMPED, END_DAY_CLAMPED):
    # Deterministic per-day initialization/training randomness.
    day_seed = MODEL_SEED + day_index
    torch.manual_seed(day_seed)
    np.random.seed(day_seed % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(day_seed)

    time_step = tgt_ds.time.isel(time=day_index)

    print(
        f"\n=== Day {day_index + 1:02d}/25 | {str(time_step.values)[:10]} ===",
        flush=True,
    )

    inp_da = tgt_ds.ssh.sel(time=time_step)
    inp_da_GS = inp_da.sel(lat=slice(28, 45.0), lon=slice(-66, -49.0))

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

    tens_inp_da = torch.from_numpy(inp_da_GS_crop.values).float().to(device)
    tens_inp_da = (tens_inp_da - mean_tgt) / std_tgt
    tens_inp_da = tens_inp_da.unsqueeze(0).unsqueeze(0)
    tens_inp_da.requires_grad_(False)

    batch = TrainingItem(input=tens_inp_da, tgt=tens_inp_da)

    a2_L1 = float(a2_results["best_L1"].isel(time=day_index).values)
    a2_L2 = float(a2_results["best_L2"].isel(time=day_index).values)
    a2_theta = float(a2_results["best_theta"].isel(time=day_index).values)
    a2_effective_logit = (
        a2_logits_ds["logits"].isel(time=day_index).values.astype(np.float32)
    )

    stored_a2_eval_mse = (
        float(a2_results["best_eval_mse"].isel(time=day_index).values)
        if "best_eval_mse" in a2_results
        else float("nan")
    )

    model = GSModelA3(time, lat, lon, device).to(device)
    model.initialize_from_a2(
        a2_effective_logit=a2_effective_logit,
        a2_L1=a2_L1,
        a2_L2=a2_L2,
        a2_theta=a2_theta,
    )

    # -------------------------------------------------------------------------
    # Baseline checkpoint: this is literally A2 because g_eta == 0.
    # -------------------------------------------------------------------------
    a2_eval = evaluate_fixed(model, batch)
    a2_expected_obs = a2_eval["expected_obs"]
    budget_limit = max(BUDGET_OBS, a2_expected_obs)

    best_state = cpu_state_dict(model)
    best_eval = dict(a2_eval)
    best_stage = "A2_baseline"
    best_iteration = 0

    print(
        f"A2 baseline | old_selection_mse={stored_a2_eval_mse:.6f} | "
        f"A3_fixed_eval_mse={a2_eval['mse']:.6f} ± {a2_eval['mse_std']:.6f} | "
        f"expected_obs={a2_expected_obs:.6f} | "
        f"L1={a2_L1:.3f} L2={a2_L2:.3f} theta={np.degrees(a2_theta):.2f}°",
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
        train_global_geometry,
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
        model.raw_L1.requires_grad_(train_global_geometry)
        model.raw_L2.requires_grad_(train_global_geometry)
        model.theta.requires_grad_(train_global_geometry)
        for p in model.warp.parameters():
            p.requires_grad_(True)

        for local_k in range(n_iters):
            global_k = global_offset + local_k
            optimizer.zero_grad(set_to_none=True)

            total_loss_report = 0.0
            total_points_report = 0.0
            total_mse_report = 0.0

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
                        weight=MAX_WEIGHT,
                    )
                    scaled_loss = mean_loss / ACCUMULATION_STEPS

                total_loss_report += mean_loss.detach() / ACCUMULATION_STEPS
                total_points_report += normalized_selected_points.detach() / ACCUMULATION_STEPS
                total_mse_report += loss_mse.detach() / ACCUMULATION_STEPS

                scaler.scale(scaled_loss).backward()

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
                budget_ok = current_eval["expected_obs"] <= budget_limit + 1e-12

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

                ratio = max(L1_list[-1], L2_list[-1]) / min(L1_list[-1], L2_list[-1])

                print(
                    f"{stage_name:13s} iter {local_k + 1:03d}/{n_iters} "
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
    # Stage 1: learn only g_eta.  A2 L1/L2/theta and sensor placement frozen.
    # -------------------------------------------------------------------------
    model.logits.requires_grad_(False)
    model.raw_L1.requires_grad_(False)
    model.raw_L2.requires_grad_(False)
    model.theta.requires_grad_(False)

    optimizer_stage1 = optim.Adam(model.warp.parameters(), lr=WARP_LR)

    updates = run_training_stage(
        stage_name="geometry_only",
        n_iters=STAGE1_ITERS,
        optimizer=optimizer_stage1,
        train_logits=False,
        train_global_geometry=False,
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

    # Joint stage starts from the best checkpoint found so far.
    model.load_state_dict(best_state)
    model.to(device)

    # -------------------------------------------------------------------------
    # Stage 2: jointly fine-tune warp + A2 global geometry + sensor logits.
    # -------------------------------------------------------------------------
    model.logits.requires_grad_(True)
    model.raw_L1.requires_grad_(True)
    model.raw_L2.requires_grad_(True)
    model.theta.requires_grad_(True)
    for p in model.warp.parameters():
        p.requires_grad_(True)

    optimizer_stage2 = optim.Adam(
        [
            {"params": model.warp.parameters(), "lr": WARP_LR},
            {"params": [model.raw_L1, model.raw_L2, model.theta], "lr": COV_LR},
            {"params": [model.logits], "lr": LOGITS_LR},
        ]
    )

    updates = run_training_stage(
        stage_name="joint",
        n_iters=STAGE2_ITERS,
        optimizer=optimizer_stage2,
        train_logits=True,
        train_global_geometry=True,
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

    # Always restore the best checkpoint.  If A3 never helps, this is the
    # exact A2 checkpoint from iteration 0.
    model.load_state_dict(best_state)
    model.to(device)

    final_eval = evaluate_fixed(model, batch)

    with torch.no_grad():
        best_L1, best_L2, best_theta = model.covariance_params()
        best_L1 = float(best_L1.cpu())
        best_L2 = float(best_L2.cpu())
        best_theta = float(best_theta.cpu())

        effective_logit = model.effective_logit()[0].detach().cpu().numpy().astype(np.float32)
        obs_prob = (
            torch.sigmoid(-model.effective_logit()[0])
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        warp_field = model.warp(model.grid_xy_norm.reshape(-1, 2)).reshape(
            lat, lon, WARP_DIM
        )
        warp_field = warp_field.detach().cpu().numpy().astype(np.float32)

    best_L_major, best_L_minor, best_theta_major = canonicalize_lengths_and_angle(
        best_L1, best_L2, best_theta
    )

    improvement_pct = (
        100.0 * (a2_eval["mse"] - final_eval["mse"]) / a2_eval["mse"]
        if a2_eval["mse"] > 0
        else 0.0
    )

    print(
        f"BEST day {day_index + 1:02d}: stage={best_stage}, iter={best_iteration}, "
        f"eval_mse {a2_eval['mse']:.6f} -> {final_eval['mse']:.6f} "
        f"({improvement_pct:+.2f}% vs A2), "
        f"expected_obs={final_eval['expected_obs']:.6f}",
        flush=True,
    )

    record = {
        "time": time_step.values,
        "mean_loss": np.asarray(mean_loss_list, dtype=np.float64),
        "mean_points": np.asarray(mean_points_list, dtype=np.float64),
        "mean_mse": np.asarray(mean_mse_list, dtype=np.float64),
        "L1": np.asarray(L1_list, dtype=np.float32),
        "L2": np.asarray(L2_list, dtype=np.float32),
        "theta": np.asarray(theta_list, dtype=np.float32),
        "a2_L1": a2_L1,
        "a2_L2": a2_L2,
        "a2_theta": a2_theta,
        "a2_eval_mse": a2_eval["mse"],
        "a2_eval_obs": a2_eval["expected_obs"],
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
        "improvement_pct_vs_A2": improvement_pct,
    }
    records.append(record)

    logits_records.append(
        xr.DataArray(
            effective_logit,
            dims=["lat", "lon"],
            coords={"lat": inp_da_GS_crop.lat, "lon": inp_da_GS_crop.lon},
            name="logits",
        )
    )

    obs_prob_records.append(
        xr.DataArray(
            obs_prob,
            dims=["lat", "lon"],
            coords={"lat": inp_da_GS_crop.lat, "lon": inp_da_GS_crop.lon},
            name="obs_prob",
        )
    )

    warp_records.append(
        xr.DataArray(
            warp_field,
            dims=["lat", "lon", "latent"],
            coords={
                "lat": inp_da_GS_crop.lat,
                "lon": inp_da_GS_crop.lon,
                "latent": np.arange(WARP_DIM),
            },
            name="warp",
        )
    )

    day_state = {
        "day_index": day_index,
        "time": str(time_step.values),
        "state_dict": cpu_state_dict(model),
        "a2_eval": a2_eval,
        "best_eval": final_eval,
        "best_stage": best_stage,
        "best_iteration": best_iteration,
        "mean_tgt": mean_tgt,
        "std_tgt": std_tgt,
        "architecture": "2-64-64-4 tanh",
        "alpha": ALPHA,
    }
    model_bundle[int(day_index)] = day_state

    # Save cumulative outputs immediately after every completed day.
    save_cumulative(
        records,
        logits_records,
        obs_prob_records,
        warp_records,
        model_bundle,
    )

    # Explicit per-day checkpoint for maximum recoverability.
    checkpoint_path = OUTPUT_DIR / f"day_{day_index + 1:02d}_best.pt"
    atomic_torch_save(day_state, checkpoint_path)

    print(
        f"Saved day {day_index + 1:02d} safely to {OUTPUT_DIR}",
        flush=True,
    )

    del model, tens_inp_da, batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


print("\nDone.")
print("Saved cumulative outputs after every completed day:")
print(f"  {OUTPUT_DIR / 'results_natl60_A3_warmstart.nc'}")
print(f"  {OUTPUT_DIR / 'logits_natl60_A3_warmstart.nc'}")
print(f"  {OUTPUT_DIR / 'obs_prob_natl60_A3_warmstart.nc'}")
print(f"  {OUTPUT_DIR / 'warp_natl60_A3_warmstart.nc'}")
print(f"  {OUTPUT_DIR / 'models_natl60_A3_warmstart.pt'}")
print(f"  {OUTPUT_DIR / 'day_XX_best.pt'}")
