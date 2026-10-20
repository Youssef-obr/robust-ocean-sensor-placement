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
# Configuration
# =============================================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Same Kaggle dataset used for the validated 25-day isotropic run.
DEFAULT_KAGGLE_DATA = (
    "/kaggle/input/datasets/youssefoubari/"
    "natl60-gulf-stream-ssh/NATL60_GulfStream_ssh.nc"
)
DEFAULT_LOCAL_DATA = (
    "C:/Users/gerv9/Desktop/gumbell/data/"
    "NATL60-CJM165-daily-reg-1_20.nc"
)

DATA_PATH = os.environ.get("NATL60_PATH", DEFAULT_KAGGLE_DATA)
if not os.path.exists(DATA_PATH) and os.path.exists(DEFAULT_LOCAL_DATA):
    DATA_PATH = DEFAULT_LOCAL_DATA

# Keep A2 outputs separate from the validated isotropic baseline.
OUTPUT_DIR = Path(
    os.environ.get(
        "GS_OUTPUT_DIR",
        str(Path(__file__).resolve().parent / "outputs_A2"),
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# EXACT training conditions from the successful memory-safe 25-day run.
n_draw = 1
n_iter = 800
rate = 0.999
budget_obs = 1 - 0.999          # 0.001 = 0.1 %
accumulation_steps = 60         # 60 x 1 draw = 4 x 15 draws per optimizer update

warmup_iterations = 100
delta_weight = 0.2
max_weight = 75.0
start_temp = 15.0
min_temp = 4.0
alpha = 0.9                     # kept from the original script
lr = 1e-1
OI_CHUNK_SIZE = 5000

# A2 initial covariance: exactly isotropic at initialization, matching L = 25.
INITIAL_L1 = 25.0
INITIAL_L2 = 25.0
INITIAL_THETA = 0.0             # radians


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

    Let delta = r_i - r_j.  We rotate delta into the principal-axis frame:
        d1 =  cos(theta) * dx + sin(theta) * dy
        d2 = -sin(theta) * dx + cos(theta) * dy

    and use
        C = sigma_f^2 * exp[-0.5 * (d1^2/L1^2 + d2^2/L2^2)].

    If L1 == L2, this reduces exactly to the isotropic RBF kernel and theta
    has no effect.
    """
    x1 = x1.unsqueeze(1)  # [N1, 1, 2]
    x2 = x2.unsqueeze(0)  # [1, N2, 2]

    delta = x1 - x2
    dx = delta[..., 0]
    dy = delta[..., 1]

    c = torch.cos(theta)
    s = torch.sin(theta)

    # R(theta)^T @ delta
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
    """
    Optimal interpolation / kriging with the A2 rotated anisotropic RBF kernel.

    Missing locations are predicted in chunks to keep GPU memory usage small.
    """
    device_ = data_with_nans.device

    observed_indices = ~torch.isnan(data_with_nans)
    missing_indices = torch.isnan(data_with_nans)

    observed_positions = torch.nonzero(observed_indices, as_tuple=False).float()
    missing_positions = torch.nonzero(missing_indices, as_tuple=False).float()

    observed_values = data_with_nans[observed_indices].float().to(device_)

    # Coordinates [x, y], identical convention to the original code.
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

    # Observed-observed covariance.
    K_oo = cov_func(observed_coords, observed_coords)
    eye = torch.eye(K_oo.size(0), device=device_, dtype=K_oo.dtype)
    K_oo = K_oo + sigma_n**2 * eye

    # Cholesky + jitter, same OI solve as baseline.
    L_chol = torch.linalg.cholesky(K_oo + 1e-6 * eye)
    alpha_oi = torch.cholesky_solve(observed_values.unsqueeze(1), L_chol)

    # Predict missing locations in chunks.
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_pred = torch.cat(pred_list, dim=0)

    reconstructed_data = data_with_nans.clone()
    reconstructed_data[missing_indices] = mean_pred.to(reconstructed_data.dtype)

    return reconstructed_data


# Non-chunked version retained for debugging / small grids only.
def optimal_interpolation(
    data_with_nans,
    L1,
    L2,
    theta,
    sigma_f=1.0,
    sigma_n=0.1,
):
    device_ = data_with_nans.device

    observed_indices = ~torch.isnan(data_with_nans)
    missing_indices = torch.isnan(data_with_nans)

    observed_positions = torch.nonzero(observed_indices, as_tuple=False).float()
    missing_positions = torch.nonzero(missing_indices, as_tuple=False).float()
    observed_values = data_with_nans[observed_indices].float().to(device_)

    observed_coords = observed_positions[:, [1, 0]].to(device_)
    missing_coords = missing_positions[:, [1, 0]].to(device_)

    L1 = _as_tensor(L1, device=device_)
    L2 = _as_tensor(L2, device=device_)
    theta = _as_tensor(theta, device=device_)
    sigma_f = _as_tensor(sigma_f, device=device_)
    sigma_n = _as_tensor(sigma_n, device=device_)

    K = _anisotropic_rbf_covariance(
        observed_coords, observed_coords, L1, L2, theta, sigma_f
    )
    eye = torch.eye(K.size(0), device=device_, dtype=K.dtype)
    K = K + sigma_n**2 * eye

    K_s = _anisotropic_rbf_covariance(
        observed_coords, missing_coords, L1, L2, theta, sigma_f
    )

    L_chol = torch.linalg.cholesky(K + 1e-6 * eye)
    alpha_oi = torch.cholesky_solve(observed_values.unsqueeze(1), L_chol)
    mean_pred = K_s.t().matmul(alpha_oi).squeeze().to(torch.float32)

    reconstructed_data = data_with_nans.clone()
    reconstructed_data[missing_indices] = mean_pred

    return reconstructed_data


# =============================================================================
# Gumbel sensor-placement model
# =============================================================================

class GSModel(nn.Module):
    def __init__(self, time, lat, lon, rate, device):
        super().__init__()

        self.logits = nn.Parameter(torch.zeros((2, time, lat, lon)))
        self.logits.data[1, :, :, :] = np.log(rate / (1 - rate))

        self.time_window = time
        self.lat = lat
        self.lon = lon
        self.device = device

        # Positive length scales via softplus.
        #
        # At values around 25, softplus(raw_L) ~= raw_L and its derivative is
        # ~= 1. This keeps the effective optimization scale close to the
        # original directly-trained scalar L while guaranteeing L1,L2 > 0.
        self.raw_L1 = nn.Parameter(torch.tensor(INITIAL_L1, dtype=torch.float32))
        self.raw_L2 = nn.Parameter(torch.tensor(INITIAL_L2, dtype=torch.float32))

        # theta is unconstrained. Periodicity is handled only when interpreting
        # / reporting the trained covariance; constraining it during training is
        # unnecessary and can introduce artificial boundaries.
        self.theta = nn.Parameter(torch.tensor(INITIAL_THETA, dtype=torch.float32))

    def covariance_params(self):
        eps = 1e-6
        L1 = F.softplus(self.raw_L1) + eps
        L2 = F.softplus(self.raw_L2) + eps
        return L1, L2, self.theta

    def gumbel_softmax(self, logits, tau=1.0, hard=False):
        gumbels = (
            -torch.empty_like(
                logits,
                memory_format=torch.legacy_contiguous_format,
            )
            .exponential_()
            .log()
        )
        gumbels = (logits + gumbels) / tau
        y_soft = gumbels.softmax(dim=0)

        if hard:
            index = y_soft.max(dim=0, keepdim=True)[1]
            y_hard = torch.zeros_like(
                logits,
                memory_format=torch.legacy_contiguous_format,
            ).scatter_(dim=0, index=index, value=1.0)
            return y_hard - y_soft.detach() + y_soft

        return y_soft

    def compute_logits_loss(
        self,
        logits_init,
        batch,
        budget_obs=0.01,
        tau=1.0,
        weight=1.0,
        hard=True,
    ):
        tgt_inp = batch.tgt.clone()
        logits = logits_init.to(self.device)

        # Same hard-forward, straight-through Gumbel-Softmax used in baseline.
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
        )
        mask_input = mask_input.unsqueeze(0)

        normalized_selected_points = mask_input.mean()

        selected_tgt_inp = tgt_inp * mask_input
        selected_tgt_inp[mask_input == 0] = float("nan")

        L1, L2, theta = self.covariance_params()

        # IMPORTANT: memory-safe chunked OI, exactly as in the successful run.
        output = optimal_interpolation_chunk(
            selected_tgt_inp[0, 0],
            L1=L1,
            L2=L2,
            theta=theta,
            chunk_size=OI_CHUNK_SIZE,
        )

        output = output.unsqueeze(0).unsqueeze(0)
        output = output.to(tgt_inp.dtype)

        loss_mse = F.mse_loss(output, batch.tgt)

        # Preserve the original budget-penalty logic exactly.
        if normalized_selected_points >= budget_obs:
            effective_weight = weight
        else:
            effective_weight = 0

        loss = loss_mse + effective_weight * normalized_selected_points

        return (
            output,
            loss,
            mask_input,
            normalized_selected_points,
            loss_mse,
        )

    def normalize_logits(self):
        norm = self.logits.norm(p=2, dim=0, keepdim=True)
        self.logits.data = self.logits.data / norm

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
# Data: same 25 dates as the successful baseline
# =============================================================================

print(f"Device: {device}")
print(f"Opening data: {DATA_PATH}")

tgt_ds = xr.open_dataset(DATA_PATH)

# One day every 15 days: indices 0, 15, ..., 360 => exactly 25 days.
tgt_ds = tgt_ds.isel(time=slice(0, None, 15))
assert tgt_ds.sizes["time"] == 25, (
    f"Expected exactly 25 days, got {tgt_ds.sizes['time']}."
)

print("Selected days:", tgt_ds.sizes["time"])
print("Training configuration:")
print(
    f"  n_iter={n_iter}, n_draw={n_draw}, "
    f"accumulation_steps={accumulation_steps}, chunk={OI_CHUNK_SIZE}"
)
print(
    f"  budget={budget_obs}, warmup={warmup_iterations}, lr={lr}"
)
print(
    f"  A2 init: L1={INITIAL_L1}, L2={INITIAL_L2}, "
    f"theta={INITIAL_THETA} rad"
)


# =============================================================================
# Result containers
# =============================================================================

all_mean_loss_list = []
all_mean_points_list = []
all_mean_mse_list = []

all_L1_list = []
all_L2_list = []
all_theta_list = []

all_time_steps = []

all_logits_list = []
all_obs_prob_list = []


# =============================================================================
# Train a fresh model independently for each selected day
# =============================================================================

for day_index, time_step in enumerate(tgt_ds.time):
    print(
        f"\n=== Day {day_index + 1:02d}/25 | "
        f"{str(time_step.values)[:10]} ===",
        flush=True,
    )

    inp_da = tgt_ds.ssh.sel(time=time_step)

    # Same Gulf Stream box as original.  If DATA_PATH already contains the
    # extracted 340x340 box, this simply selects the full available box.
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

    assert (lat, lon) == (280, 280), (
        f"Expected final 280x280 crop, got {(lat, lon)}."
    )

    # Fresh model per day, same as baseline.
    model = GSModel(time, lat, lon, rate, device).to(device)

    mean_tgt = inp_da_GS_crop.mean().item()
    std_tgt = inp_da_GS_crop.std().item()

    TrainingItem = namedtuple("TrainingItem", ["input", "tgt"])

    tens_inp_da = torch.from_numpy(
        inp_da_GS_crop.values
    ).float().to(device)

    tens_inp_da = (tens_inp_da - mean_tgt) / std_tgt
    tens_inp_da = tens_inp_da.unsqueeze(0).unsqueeze(0)
    tens_inp_da.requires_grad_(True)

    batch = TrainingItem(
        input=tens_inp_da,
        tgt=tens_inp_da,
    )

    temperature = start_temp

    mean_loss_list = []
    mean_points_list = []
    mean_mse_list = []

    L1_list = []
    L2_list = []
    theta_list = []

    # Kept from the original code.  The update below intentionally preserves
    # the same temperature behavior as the validated baseline run.
    decay_rate = (min_temp / start_temp) ** (1 / (n_iter - 1))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    for k in range(n_iter):
        optimizer.zero_grad(set_to_none=True)

        total_loss_report = 0.0
        total_points_report = 0.0
        total_mse_report = 0.0

        if k < warmup_iterations:
            weight = 0
        else:
            weight = min(
                (k - warmup_iterations + 1) * delta_weight,
                max_weight,
            )

        # Memory-safe equivalent of the original 4 x 15 draws:
        # 60 microsteps x 1 draw, then one optimizer update.
        for _ in range(accumulation_steps):
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
                    budget_obs=budget_obs,
                    tau=temperature,
                    n_draw=n_draw,
                    weight=weight,
                )

                scaled_loss = mean_loss / accumulation_steps

            # Reporting tensors are detached so old OI graphs are not retained.
            total_loss_report += (
                mean_loss.detach() / accumulation_steps
            )
            total_points_report += (
                normalized_selected_points.detach() / accumulation_steps
            )
            total_mse_report += (
                loss_mse.detach() / accumulation_steps
            )

            scaler.scale(scaled_loss).backward()

        # At initialization L1 == L2, so the covariance is isotropic and theta
        # is mathematically unidentifiable. Tiny numerical/AMP noise can still
        # produce a non-zero theta gradient. Freeze theta for the first optimizer
        # update only; from k >= 1 it is learned normally once anisotropy can emerge.
        if k == 0:
            model.theta.grad = None

        scaler.step(optimizer)
        scaler.update()

        # Current covariance values AFTER the optimizer update.
        with torch.no_grad():
            L1_now, L2_now, theta_now = model.covariance_params()

        mean_loss_list.append(float(total_loss_report.cpu()))
        mean_points_list.append(float(total_points_report.cpu()))
        mean_mse_list.append(float(total_mse_report.cpu()))

        L1_list.append(float(L1_now.detach().cpu()))
        L2_list.append(float(L2_now.detach().cpu()))
        theta_list.append(float(theta_now.detach().cpu()))

        # IMPORTANT: preserve the same near-constant temperature update used in
        # the successful baseline; do not silently "fix" the original behavior.
        temperature = start_temp * (
            min_temp / start_temp
        ) ** (1 / (n_iter - 1))

        if (k + 1) % 50 == 0 or k == 0:
            ratio = max(
                float(L1_now.detach().cpu()),
                float(L2_now.detach().cpu()),
            ) / min(
                float(L1_now.detach().cpu()),
                float(L2_now.detach().cpu()),
            )

            print(
                f"iter {k + 1:03d}/{n_iter} | "
                f"loss={mean_loss_list[-1]:.6f} | "
                f"mse={mean_mse_list[-1]:.6f} | "
                f"obs={mean_points_list[-1]:.6f} | "
                f"L1={L1_list[-1]:.3f} | "
                f"L2={L2_list[-1]:.3f} | "
                f"ratio={ratio:.3f} | "
                f"theta={theta_list[-1]:.4f} rad "
                f"({np.degrees(theta_list[-1]):.2f} deg)",
                flush=True,
            )

    # -------------------------------------------------------------------------
    # Store this day's training history
    # -------------------------------------------------------------------------

    all_mean_loss_list.append(mean_loss_list)
    all_mean_points_list.append(mean_points_list)
    all_mean_mse_list.append(mean_mse_list)

    all_L1_list.append(L1_list)
    all_L2_list.append(L2_list)
    all_theta_list.append(theta_list)

    all_time_steps.append(time_step.values)

    # Same saved logits / implied observation probability as baseline.
    logits_da = xr.DataArray(
        model.logits[1, 0, :, :].detach().cpu().numpy(),
        dims=["lat", "lon"],
        coords={
            "lat": inp_da_GS_crop.lat,
            "lon": inp_da_GS_crop.lon,
        },
        name="logits",
    )

    obs_prob_da = xr.DataArray(
        1
        - 1
        / (
            1
            + np.exp(
                -model.logits[1, 0, :, :]
                .detach()
                .cpu()
                .numpy()
            )
        ),
        dims=["lat", "lon"],
        coords={
            "lat": inp_da_GS_crop.lat,
            "lon": inp_da_GS_crop.lon,
        },
        name="obs_prob",
    )

    all_logits_list.append(logits_da)
    all_obs_prob_list.append(obs_prob_da)


# =============================================================================
# Save A2 results
# =============================================================================

coords = {
    "time": all_time_steps,
    "iteration": np.arange(n_iter),
}

mean_loss_da = xr.DataArray(
    all_mean_loss_list,
    dims=["time", "iteration"],
    coords=coords,
)

mean_points_da = xr.DataArray(
    all_mean_points_list,
    dims=["time", "iteration"],
    coords=coords,
)

mean_mse_da = xr.DataArray(
    all_mean_mse_list,
    dims=["time", "iteration"],
    coords=coords,
)

L1_da = xr.DataArray(
    all_L1_list,
    dims=["time", "iteration"],
    coords=coords,
)

L2_da = xr.DataArray(
    all_L2_list,
    dims=["time", "iteration"],
    coords=coords,
)

theta_da = xr.DataArray(
    all_theta_list,
    dims=["time", "iteration"],
    coords=coords,
)

results_ds = xr.Dataset(
    {
        "mean_loss": mean_loss_da,
        "mean_points": mean_points_da,
        "mean_mse": mean_mse_da,
        "L1": L1_da,
        "L2": L2_da,
        "theta": theta_da,
    }
)

results_ds["L1"].attrs["units"] = "grid cells"
results_ds["L2"].attrs["units"] = "grid cells"
results_ds["theta"].attrs["units"] = "radians"
results_ds.attrs["covariance"] = "rotated anisotropic RBF (A2)"
results_ds.attrs["n_days"] = 25
results_ds.attrs["day_stride"] = 15
results_ds.attrs["n_iter"] = n_iter
results_ds.attrs["n_draw"] = n_draw
results_ds.attrs["accumulation_steps"] = accumulation_steps
results_ds.attrs["oi_chunk_size"] = OI_CHUNK_SIZE
results_ds.attrs["budget_obs"] = budget_obs

results_path = OUTPUT_DIR / "results_natl60_A2.nc"
logits_path = OUTPUT_DIR / "logits_natl60_A2.nc"
obs_prob_path = OUTPUT_DIR / "obs_prob_natl60_A2.nc"

results_ds.to_netcdf(results_path)

logits_da_all = xr.concat(
    all_logits_list,
    dim=xr.DataArray(
        all_time_steps,
        dims="time",
        name="time",
    ),
)
obs_prob_da_all = xr.concat(
    all_obs_prob_list,
    dim=xr.DataArray(
        all_time_steps,
        dims="time",
        name="time",
    ),
)

logits_da_all.to_netcdf(logits_path)
obs_prob_da_all.to_netcdf(obs_prob_path)

print("\nDone.")
print("Saved:")
print(f"  {results_path}")
print(f"  {logits_path}")
print(f"  {obs_prob_path}")
