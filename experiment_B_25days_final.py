import os
import math
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import xarray as xr
from torch.amp import autocast, GradScaler


# =============================================================================
# Experiment B -- robustness / error attribution for sensor placement
# =============================================================================
#
# 25 NATL60 days (same every-15-days subset as Experiment A)
# 6 error families x 3 severities + clean = 19 conditions per day
# 10 forecast-ensemble members per condition
#
# For every (day, condition):
#   imperfect forecast ensemble
#        -> train Isotropic
#        -> warm-start/train A2
#        -> warm-start/train A3
#        -> freeze
#        -> observe the CLEAN truth with the learned sensor logits
#        -> reconstruct CLEAN truth
#        -> 100 fresh hard Gumbel masks for physical RMSE / EV / sensors
#
# A2 is an internal bridge only; the final summary contains:
#   A3, Isotropic, Random
#
# Strict output policy: only three cumulative files are written:
#   1) experiment_B_daily.csv   : per-day raw metrics / resume ledger
#   2) experiment_B_summary.csv : final compact aggregate table
#   3) experiment_B_states.pt   : minimal consolidated state needed for later UQ
#
# No per-day checkpoints, logits files, warp grids, or training histories are
# created.  The state file stores only effective logits + covariance geometry
# (and the small A3 warp-network weights) for Isotropic and A3.
# =============================================================================


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

OUTPUT_DIR = Path(os.environ.get("GS_OUTPUT_DIR", str(SCRIPT_DIR / "outputs_B")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DAILY_CSV = OUTPUT_DIR / "experiment_B_daily.csv"
SUMMARY_CSV = OUTPUT_DIR / "experiment_B_summary.csv"
STATES_PT = OUTPUT_DIR / "experiment_B_states.pt"
STATE_FORMAT_VERSION = 1

# Same 25-day sampling as Experiment A.
START_DAY = int(os.environ.get("GS_START_DAY", "0"))
END_DAY = int(os.environ.get("GS_END_DAY", "25"))

# Forecast ensemble.
ENSEMBLE_SIZE = 10

# Sensor budget / OI settings.
RATE = 0.999
BUDGET_OBS = 1.0 - RATE
MAX_WEIGHT = 75.0
SIGMA_F = 1.0
SIGMA_N = 0.1
OI_CHUNK_SIZE = int(os.environ.get("B_OI_CHUNK_SIZE", "5000"))

# Preserve the validated Experiment-A near-constant Gumbel temperature behavior.
START_TEMP = 15.0
MIN_TEMP = 4.0
BASELINE_REFERENCE_ITERS = 800
TRAIN_TEMP = START_TEMP * (MIN_TEMP / START_TEMP) ** (1.0 / (BASELINE_REFERENCE_ITERS - 1))

# Isotropic training: same validated baseline recipe.
ISO_ITERS = int(os.environ.get("B_ISO_ITERS", "800"))
ISO_LR = float(os.environ.get("B_ISO_LR", "0.1"))
ISO_WARMUP = int(os.environ.get("B_ISO_WARMUP", "100"))
ISO_DELTA_WEIGHT = float(os.environ.get("B_ISO_DELTA_WEIGHT", "0.2"))

# A2 warm start: same setup used in Experiment A.
A2_STAGE1_ITERS = int(os.environ.get("B_A2_STAGE1_ITERS", "150"))
A2_STAGE2_ITERS = int(os.environ.get("B_A2_STAGE2_ITERS", "250"))
A2_COV_LR = float(os.environ.get("B_A2_COV_LR", "0.02"))
A2_LOGITS_LR = float(os.environ.get("B_A2_LOGITS_LR", "0.01"))
A2_EVAL_EVERY = int(os.environ.get("B_A2_EVAL_EVERY", "50"))
A2_EVAL_DRAWS = int(os.environ.get("B_A2_EVAL_DRAWS", "16"))
A2_EVAL_SEED = int(os.environ.get("B_A2_EVAL_SEED", "271828"))

# A3 warm start: same 150 + 250 recipe that produced the reported A3 result.
A3_STAGE1_ITERS = int(os.environ.get("B_A3_STAGE1_ITERS", "150"))
A3_STAGE2_ITERS = int(os.environ.get("B_A3_STAGE2_ITERS", "250"))
A3_WARP_LR = float(os.environ.get("B_A3_WARP_LR", "0.005"))
A3_COV_LR = float(os.environ.get("B_A3_COV_LR", "0.005"))
A3_LOGITS_LR = float(os.environ.get("B_A3_LOGITS_LR", "0.005"))
A3_EVAL_EVERY = int(os.environ.get("B_A3_EVAL_EVERY", "50"))
A3_EVAL_DRAWS = int(os.environ.get("B_A3_EVAL_DRAWS", "16"))
A3_EVAL_SEED = int(os.environ.get("B_A3_EVAL_SEED", "314159"))

# Memory-safe stochastic accumulation.  Because 60 is divisible by 10, each
# optimizer step sees every forecast-ensemble member exactly six times, but we
# only run one member/OI solve at a time.  This estimates the ensemble-average
# objective without multiplying GPU memory or OI work by another factor of 10.
ACCUMULATION_STEPS = int(os.environ.get("B_ACCUMULATION_STEPS", "60"))
if ACCUMULATION_STEPS % ENSEMBLE_SIZE != 0:
    raise ValueError(
        f"B_ACCUMULATION_STEPS={ACCUMULATION_STEPS} must be divisible by "
        f"ENSEMBLE_SIZE={ENSEMBLE_SIZE}."
    )

# Final physical evaluation.
FINAL_EVAL_DRAWS = int(os.environ.get("B_FINAL_EVAL_DRAWS", "100"))
FINAL_EVAL_SEED = int(os.environ.get("B_FINAL_EVAL_SEED", "123456"))

# A3 architecture.
WARP_HIDDEN = 64
WARP_DIM = 4
ALPHA = 1.0

# Model / condition reproducibility.
MODEL_SEED = int(os.environ.get("B_MODEL_SEED", "424242"))
PERTURB_SEED = int(os.environ.get("B_PERTURB_SEED", "20261014"))

# Paper warped experiment: medium deformation is EXACTLY the provided setting
# in terms of displacement generator and warp operator.  Mild / strong simply
# scale those exact displacement fields before warping.
DEFORMATION_FACTORS = {
    "mild": 0.5,
    "medium": 1.0,
    "strong": 1.5,
}
PAPER_DISP_L = 5.0
PAPER_DISP_TCORR = 1.0
PAPER_DISP_SIGMA = 35.0
PAPER_DISP_SEED_X = 41
PAPER_DISP_SEED_Y = 42

ERROR_TYPES = [
    "deformation",
    "amplitude",
    "large_scale_bias",
    "noise",
    "smoothing",
    "combined",
]
SEVERITIES = ["mild", "medium", "strong"]
CONDITIONS = [("clean", "clean")] + [
    (error_type, severity)
    for error_type in ERROR_TYPES
    for severity in SEVERITIES
]

METHOD_ORDER = {"A3": 0, "Isotropic": 1, "Random": 2}
ERROR_ORDER = {"clean": 0, **{name: i + 1 for i, name in enumerate(ERROR_TYPES)}}
SEVERITY_ORDER = {"clean": 0, "mild": 1, "medium": 2, "strong": 3}


# =============================================================================
# Generic helpers
# =============================================================================


def stable_int(*parts):
    """Deterministic 31-bit integer from arbitrary labels."""
    msg = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(msg).hexdigest()[:8], 16) & 0x7FFFFFFF


def set_all_seeds(seed):
    seed = int(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_write_csv(df, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _as_tensor(value, *, device, dtype=torch.float32):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, dtype=dtype, device=device)


def inverse_softplus(y):
    y = torch.as_tensor(y, dtype=torch.float32)
    return y + torch.log(-torch.expm1(-y))


def rms(x):
    return torch.sqrt(torch.mean(x.float() ** 2))


def memberwise_unit_rms(delta, eps=1e-12):
    """Normalize [J,H,W] templates to unit RMS member-by-member."""
    denom = torch.sqrt(torch.mean(delta.float() ** 2, dim=(-2, -1), keepdim=True)).clamp_min(eps)
    return delta / denom.to(delta.dtype)


def mean_ensemble_rmse(ensemble, clean):
    d = ensemble - clean.unsqueeze(0)
    vals = torch.sqrt(torch.mean(d.float() ** 2, dim=(-2, -1)))
    return float(vals.mean().detach().cpu())


def expected_observation_fraction(logits):
    return torch.softmax(logits, dim=0)[0].mean()


def effective_logit(logits):
    return logits[1] - logits[0]


def hard_mask_from_fixed_seed(logits, seed):
    gen = torch.Generator(device=logits.device)
    gen.manual_seed(int(seed))
    u = torch.rand(
        logits.shape,
        dtype=logits.dtype,
        device=logits.device,
        generator=gen,
    ).clamp_(1e-7, 1.0 - 1e-7)
    gumbel = -torch.log(-torch.log(u))
    cls = torch.argmax(logits + gumbel, dim=0)
    return (cls == 0).to(logits.dtype)


def cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def condition_seed(day_index, error_type, severity, suffix):
    return MODEL_SEED + stable_int(day_index, error_type, severity, suffix) % 1_000_000_000


# =============================================================================
# OI / covariance in an arbitrary learned embedding
# =============================================================================


def embedded_rbf_covariance(z1, z2, sigma_f=1.0):
    delta = z1.unsqueeze(1) - z2.unsqueeze(0)
    sqdist = (delta * delta).sum(dim=-1)
    sigma_f = _as_tensor(sigma_f, device=z1.device, dtype=z1.dtype)
    return sigma_f**2 * torch.exp(-0.5 * sqdist)


def optimal_interpolation_embedding(
    data_with_nans,
    phi_grid,
    sigma_f=SIGMA_F,
    sigma_n=SIGMA_N,
    chunk_size=OI_CHUNK_SIZE,
):
    device_ = data_with_nans.device

    observed_indices = ~torch.isnan(data_with_nans)
    missing_indices = torch.isnan(data_with_nans)

    observed_values = data_with_nans[observed_indices].float().to(device_)
    observed_phi = phi_grid[observed_indices]
    missing_phi = phi_grid[missing_indices]

    if observed_values.numel() == 0:
        # Extremely unlikely with ~78 expected sensors, but keep the evaluation
        # numerically defined rather than crashing.
        reconstructed = data_with_nans.clone()
        reconstructed[missing_indices] = 0.0
        return reconstructed

    sigma_f_t = _as_tensor(sigma_f, device=device_, dtype=observed_phi.dtype)
    sigma_n_t = _as_tensor(sigma_n, device=device_, dtype=observed_phi.dtype)

    K_oo = embedded_rbf_covariance(observed_phi, observed_phi, sigma_f_t)
    eye = torch.eye(K_oo.size(0), device=device_, dtype=K_oo.dtype)
    K_oo = K_oo + sigma_n_t**2 * eye

    L_chol = torch.linalg.cholesky(K_oo + 1e-6 * eye)
    alpha_oi = torch.cholesky_solve(observed_values.unsqueeze(1), L_chol)

    pred_list = []
    for start in range(0, missing_phi.size(0), chunk_size):
        end = min(start + chunk_size, missing_phi.size(0))
        K_s = embedded_rbf_covariance(observed_phi, missing_phi[start:end], sigma_f_t)
        pred = K_s.t().matmul(alpha_oi).squeeze(1)
        pred_list.append(pred)
        del K_s, pred

    reconstructed = data_with_nans.clone()
    if pred_list:
        reconstructed[missing_indices] = torch.cat(pred_list, dim=0).to(reconstructed.dtype)
    return reconstructed


# =============================================================================
# Sensor-placement models
# =============================================================================


class SensorModelBase(nn.Module):
    def __init__(self, lat, lon, device):
        super().__init__()
        self.lat = int(lat)
        self.lon = int(lon)
        self.device = device
        self.logits = nn.Parameter(torch.zeros((2, 1, lat, lon), dtype=torch.float32))
        with torch.no_grad():
            self.logits[1].fill_(math.log(RATE / (1.0 - RATE)))

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

    def expected_observation_fraction(self):
        return expected_observation_fraction(self.logits)

    def effective_logit(self):
        return effective_logit(self.logits)

    def geometry_embedding(self):
        raise NotImplementedError

    def loss_on_field(self, field_2d, tau, weight):
        gs_output = F.gumbel_softmax(self.logits, tau=tau, hard=True, dim=0)[0, 0]
        obs_fraction = gs_output.mean()

        selected = field_2d * gs_output
        selected = selected.clone()
        selected[gs_output == 0] = float("nan")

        phi_grid = self.geometry_embedding()
        output = optimal_interpolation_embedding(selected, phi_grid)
        mse = F.mse_loss(output, field_2d)

        effective_weight = weight if obs_fraction >= BUDGET_OBS else 0.0
        loss = mse + effective_weight * obs_fraction
        return loss, mse, obs_fraction


class IsotropicModel(SensorModelBase):
    def __init__(self, lat, lon, device):
        super().__init__(lat, lon, device)
        self.length_scale = nn.Parameter(torch.tensor(25.0, dtype=torch.float32))

    def geometry_embedding(self):
        L = self.length_scale.clamp_min(1e-3)
        return self.grid_xy / L


class A2Model(SensorModelBase):
    def __init__(self, lat, lon, device):
        super().__init__(lat, lon, device)
        self.raw_L1 = nn.Parameter(torch.tensor(25.0, dtype=torch.float32))
        self.raw_L2 = nn.Parameter(torch.tensor(25.0, dtype=torch.float32))
        self.theta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def covariance_params(self):
        eps = 1e-6
        L1 = F.softplus(self.raw_L1) + eps
        L2 = F.softplus(self.raw_L2) + eps
        return L1, L2, self.theta

    @torch.no_grad()
    def initialize_from_isotropic(self, iso_model):
        self.logits.zero_()
        self.logits[1].copy_(iso_model.effective_logit())
        L0 = float(iso_model.length_scale.detach().clamp_min(1e-3).cpu())
        target = torch.tensor(L0 - 1e-6, dtype=torch.float32).clamp_min(1e-6)
        raw = inverse_softplus(target)
        self.raw_L1.copy_(raw.to(self.raw_L1.device))
        self.raw_L2.copy_(raw.to(self.raw_L2.device))
        self.theta.zero_()

    def geometry_embedding(self):
        L1, L2, theta = self.covariance_params()
        x = self.grid_xy[..., 0]
        y = self.grid_xy[..., 1]
        c = torch.cos(theta)
        s = torch.sin(theta)
        z1 = (c * x + s * y) / L1
        z2 = (-s * x + c * y) / L2
        return torch.stack([z1, z2], dim=-1)


class SpatialWarp(nn.Module):
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


class A3Model(SensorModelBase):
    def __init__(self, lat, lon, device):
        super().__init__(lat, lon, device)
        self.raw_L1 = nn.Parameter(torch.tensor(25.0, dtype=torch.float32))
        self.raw_L2 = nn.Parameter(torch.tensor(25.0, dtype=torch.float32))
        self.theta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.warp = SpatialWarp()

    def covariance_params(self):
        eps = 1e-6
        L1 = F.softplus(self.raw_L1) + eps
        L2 = F.softplus(self.raw_L2) + eps
        return L1, L2, self.theta

    @torch.no_grad()
    def initialize_from_a2(self, a2_model):
        self.logits.zero_()
        self.logits[1].copy_(a2_model.effective_logit())

        a2_L1, a2_L2, a2_theta = a2_model.covariance_params()
        target_L1 = (a2_L1.detach().cpu() - 1e-6).clamp_min(1e-6)
        target_L2 = (a2_L2.detach().cpu() - 1e-6).clamp_min(1e-6)
        self.raw_L1.copy_(inverse_softplus(target_L1).to(self.raw_L1.device))
        self.raw_L2.copy_(inverse_softplus(target_L2).to(self.raw_L2.device))
        self.theta.copy_(a2_theta.detach().to(self.theta.device))

        # Exact A2 start.
        nn.init.zeros_(self.warp.fc3.weight)
        nn.init.zeros_(self.warp.fc3.bias)

    def geometry_embedding(self):
        L1, L2, theta = self.covariance_params()
        x = self.grid_xy[..., 0]
        y = self.grid_xy[..., 1]
        c = torch.cos(theta)
        s = torch.sin(theta)
        z1 = (c * x + s * y) / L1
        z2 = (-s * x + c * y) / L2
        zeros = torch.zeros_like(z1)
        base4 = torch.stack([z1, z2, zeros, zeros], dim=-1)

        g = self.warp(self.grid_xy_norm.reshape(-1, 2)).reshape(self.lat, self.lon, WARP_DIM)
        return base4 + ALPHA * g


# =============================================================================
# Paper-style deformation generator
# =============================================================================


def generate_correlated_fields(N, L, T_corr, sigma, num_fields=10, device=DEVICE, seed=41):
    # This follows the provided warped repository code.
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    time_points = torch.linspace(0, num_fields - 1, num_fields, device=device)
    C_temporal = torch.exp(-torch.abs(time_points[:, None] - time_points[None, :]) / T_corr)
    L_chol = torch.linalg.cholesky(C_temporal)

    white_noises = torch.randn((num_fields, N, N), device=device, generator=gen)
    temporal_correlated_noises = torch.matmul(
        L_chol, white_noises.view(num_fields, -1)
    ).view(num_fields, N, N)

    kx = torch.fft.fftfreq(N, device=device) * N
    ky = torch.fft.fftfreq(N, device=device) * N
    k = torch.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    cutoff_mask = (k < 20).float()

    P_k = torch.exp(-0.5 * (k * L) ** 3)
    P_k[0, 0] = 0.0
    P_k = P_k / torch.sum(P_k)

    fields = []
    for i in range(num_fields):
        noise_ft = torch.fft.fft2(temporal_correlated_noises[i])
        field_ft = noise_ft * sigma**2 * torch.sqrt(P_k) * cutoff_mask
        fields.append(torch.fft.ifft2(field_ft).real)
    return torch.stack(fields)


def warp_field_repo_exact(field, dx, dy):
    """Repository warp convention, including periodic coordinate wrapping."""
    batch_size, _, height, width = field.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=field.device),
        torch.arange(width, device=field.device),
        indexing="ij",
    )
    base_grid = torch.stack((x, y), dim=-1).float()
    base_grid = base_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    sample_grid = base_grid + torch.stack((dx.to(field.dtype), dy.to(field.dtype)), dim=-1)
    sample_grid[..., 0] = sample_grid[..., 0] % width
    sample_grid[..., 1] = sample_grid[..., 1] % height

    sample_grid[..., 0] = 2.0 * sample_grid[..., 0] / width - 1.0
    sample_grid[..., 1] = 2.0 * sample_grid[..., 1] / height - 1.0

    return F.grid_sample(
        field,
        sample_grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )


# =============================================================================
# Controlled perturbations
# =============================================================================


def gaussian_kernel_2d(kernel_size, sigma, device, dtype):
    x = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def gaussian_blur_batch(fields, sigma):
    """fields: [J,H,W] or [1,H,W]; reflect-padded Gaussian blur."""
    if sigma <= 0:
        return fields
    kernel_size = int(max(5, 2 * math.ceil(3.0 * float(sigma)) + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    k = gaussian_kernel_2d(kernel_size, float(sigma), fields.device, fields.dtype)
    x = fields.unsqueeze(1)
    pad = kernel_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x, k).squeeze(1)


def low_frequency_templates(num_members, H, W, seed):
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(int(seed))
    white = torch.randn((num_members, H, W), device=DEVICE, generator=gen)
    smooth = gaussian_blur_batch(white, sigma=18.0)
    smooth = smooth - smooth.mean(dim=(-2, -1), keepdim=True)
    return memberwise_unit_rms(smooth)


def white_noise_templates(num_members, H, W, seed):
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(int(seed))
    noise = torch.randn((num_members, H, W), device=DEVICE, generator=gen)
    noise = noise - noise.mean(dim=(-2, -1), keepdim=True)
    return memberwise_unit_rms(noise)


def amplitude_templates(clean_crop):
    # Global too-strong / too-weak amplitude.  Each member has unit-RMS delta.
    clean_unit = clean_crop / rms(clean_crop).clamp_min(1e-12)
    signs = torch.tensor(
        [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        device=clean_crop.device,
        dtype=clean_crop.dtype,
    )[:, None, None]
    return signs * clean_unit.unsqueeze(0)


def smoothing_templates(clean_crop, severity):
    base_sigma = {"mild": 1.5, "medium": 3.0, "strong": 6.0}[severity]
    multipliers = torch.linspace(0.85, 1.15, ENSEMBLE_SIZE, device=clean_crop.device)
    templates = []
    for mult in multipliers:
        blurred = gaussian_blur_batch(clean_crop.unsqueeze(0), sigma=base_sigma * float(mult))[0]
        templates.append(blurred - clean_crop)
    return memberwise_unit_rms(torch.stack(templates))


def build_all_forecast_ensembles(
    clean_full_norm,
    clean_crop_norm,
    border_size,
    day_index,
    base_dx,
    base_dy,
):
    """
    Build all 19 forecast ensembles for one day.

    Fair-severity rule:
      1) deformation severity is defined by the repository displacement fields;
      2) the actual mean deformation RMSE for that day/severity becomes the
         target forecast RMSE for amplitude/bias/noise/smoothing/combined.

    Therefore different error TYPES are compared at approximately equal input
    forecast-error magnitude, while medium deformation remains the repo setting.
    """
    H, W = clean_crop_norm.shape
    clean_ens = clean_crop_norm.unsqueeze(0).repeat(ENSEMBLE_SIZE, 1, 1)

    out = {("clean", "clean"): clean_ens}
    deformation_ensembles = {}
    target_rmse = {}

    full_batch = clean_full_norm.unsqueeze(0).unsqueeze(1).repeat(ENSEMBLE_SIZE, 1, 1, 1)

    for severity in SEVERITIES:
        factor = DEFORMATION_FACTORS[severity]
        warped_full = warp_field_repo_exact(full_batch, factor * base_dx, factor * base_dy)
        warped_crop = warped_full[:, 0, border_size:-border_size, border_size:-border_size]
        deformation_ensembles[severity] = warped_crop
        target_rmse[severity] = mean_ensemble_rmse(warped_crop, clean_crop_norm)
        out[("deformation", severity)] = warped_crop

    # Day-specific stochastic templates, fixed across severity levels.
    seed_base = PERTURB_SEED + 1000 * day_index
    bias_unit = low_frequency_templates(
        ENSEMBLE_SIZE, H, W, seed=seed_base + 101
    ).to(clean_crop_norm.dtype)
    noise_unit = white_noise_templates(
        ENSEMBLE_SIZE, H, W, seed=seed_base + 202
    ).to(clean_crop_norm.dtype)
    amp_unit = amplitude_templates(clean_crop_norm)

    for severity in SEVERITIES:
        target = target_rmse[severity]
        target_t = torch.tensor(target, device=clean_crop_norm.device, dtype=clean_crop_norm.dtype)

        amp_ens = clean_ens + target_t * amp_unit
        bias_ens = clean_ens + target_t * bias_unit
        noise_ens = clean_ens + target_t * noise_unit

        smooth_unit = smoothing_templates(clean_crop_norm, severity)
        smooth_ens = clean_ens + target_t * smooth_unit

        out[("amplitude", severity)] = amp_ens
        out[("large_scale_bias", severity)] = bias_ens
        out[("noise", severity)] = noise_ens
        out[("smoothing", severity)] = smooth_ens

        # Combined condition: equal-energy contribution from all five basic
        # error families, then renormalize the total perturbation to the same
        # target RMSE so severity stays comparable across all six families.
        d_def = deformation_ensembles[severity] - clean_ens
        d_amp = amp_ens - clean_ens
        d_bias = bias_ens - clean_ens
        d_noise = noise_ens - clean_ens
        d_smooth = smooth_ens - clean_ens

        combined_delta = (
            memberwise_unit_rms(d_def)
            + memberwise_unit_rms(d_amp)
            + memberwise_unit_rms(d_bias)
            + memberwise_unit_rms(d_noise)
            + memberwise_unit_rms(d_smooth)
        ) / math.sqrt(5.0)
        combined_delta = memberwise_unit_rms(combined_delta)
        out[("combined", severity)] = clean_ens + target_t * combined_delta

    return out


# =============================================================================
# Training / checkpoint selection
# =============================================================================


def train_isotropic(model, forecast_ensemble):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=ISO_LR)
    scaler = GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    for k in range(ISO_ITERS):
        optimizer.zero_grad(set_to_none=True)

        if k < ISO_WARMUP:
            weight = 0.0
        else:
            weight = min((k - ISO_WARMUP + 1) * ISO_DELTA_WEIGHT, MAX_WEIGHT)

        report_mse = 0.0
        report_obs = 0.0
        for acc in range(ACCUMULATION_STEPS):
            member = acc % ENSEMBLE_SIZE
            with autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                loss, mse, obs = model.loss_on_field(
                    forecast_ensemble[member], tau=TRAIN_TEMP, weight=weight
                )
                scaled = loss / ACCUMULATION_STEPS
            scaler.scale(scaled).backward()
            report_mse += float(mse.detach().cpu()) / ACCUMULATION_STEPS
            report_obs += float(obs.detach().cpu()) / ACCUMULATION_STEPS

        scaler.step(optimizer)
        scaler.update()

        if (k + 1) % 100 == 0 or (k + 1) == ISO_ITERS:
            print(
                f"    Isotropic {k+1:04d}/{ISO_ITERS} | "
                f"forecast_mse={report_mse:.6f} | obs={report_obs:.6f} | "
                f"L={float(model.length_scale.detach().cpu()):.3f}",
                flush=True,
            )

    return model


@torch.no_grad()
def evaluate_forecast_fixed(model, forecast_ensemble, seed, draws):
    model.eval()
    phi_grid = model.geometry_embedding()
    mse_values = []

    for j in range(draws):
        member = j % ENSEMBLE_SIZE
        mask = hard_mask_from_fixed_seed(model.logits, seed + j)[0]
        selected = forecast_ensemble[member] * mask
        selected = selected.clone()
        selected[mask == 0] = float("nan")
        output = optimal_interpolation_embedding(selected, phi_grid)
        mse_values.append(float(F.mse_loss(output, forecast_ensemble[member]).cpu()))

    return {
        "mse": float(np.mean(mse_values)),
        "expected_obs": float(model.expected_observation_fraction().cpu()),
    }


def run_stage_with_checkpointing(
    model,
    forecast_ensemble,
    optimizer,
    n_iters,
    eval_every,
    eval_draws,
    eval_seed,
    budget_limit,
    best_state,
    best_eval,
    stage_name,
):
    scaler = GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    for k in range(n_iters):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        report_mse = 0.0
        report_obs = 0.0

        for acc in range(ACCUMULATION_STEPS):
            member = acc % ENSEMBLE_SIZE
            with autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                loss, mse, obs = model.loss_on_field(
                    forecast_ensemble[member], tau=TRAIN_TEMP, weight=MAX_WEIGHT
                )
                scaled = loss / ACCUMULATION_STEPS
            scaler.scale(scaled).backward()
            report_mse += float(mse.detach().cpu()) / ACCUMULATION_STEPS
            report_obs += float(obs.detach().cpu()) / ACCUMULATION_STEPS

        # At exact A2 isotropy theta is initially unidentifiable.  Match the
        # validated A2 warm-start behavior by leaving theta untouched on the
        # very first covariance-only update; once L1/L2 separate, theta becomes
        # identifiable naturally.
        if stage_name == "A2 geometry" and k == 0:
            model.theta.grad = None

        scaler.step(optimizer)
        scaler.update()

        should_eval = ((k + 1) % eval_every == 0) or ((k + 1) == n_iters)
        if should_eval:
            current = evaluate_forecast_fixed(
                model, forecast_ensemble, seed=eval_seed, draws=eval_draws
            )
            budget_ok = current["expected_obs"] <= budget_limit + 1e-12
            improved = budget_ok and current["mse"] < best_eval["mse"]
            if improved:
                best_state = cpu_state_dict(model)
                best_eval = dict(current)

            status = " <-- NEW BEST" if improved else (" [over budget]" if not budget_ok else "")
            print(
                f"      {stage_name:16s} {k+1:03d}/{n_iters} | "
                f"train_mse={report_mse:.6f} | eval_mse={current['mse']:.6f} | "
                f"exp_obs={current['expected_obs']:.6f}{status}",
                flush=True,
            )

    return best_state, best_eval


def train_a2_from_isotropic(iso_model, forecast_ensemble, lat, lon):
    a2 = A2Model(lat, lon, DEVICE).to(DEVICE)
    a2.initialize_from_isotropic(iso_model)

    baseline = evaluate_forecast_fixed(
        a2, forecast_ensemble, seed=A2_EVAL_SEED, draws=A2_EVAL_DRAWS
    )
    budget_limit = max(BUDGET_OBS, baseline["expected_obs"])
    best_state = cpu_state_dict(a2)
    best_eval = dict(baseline)

    # Stage 1: covariance geometry only.
    a2.logits.requires_grad_(False)
    a2.raw_L1.requires_grad_(True)
    a2.raw_L2.requires_grad_(True)
    a2.theta.requires_grad_(True)
    opt1 = optim.Adam([a2.raw_L1, a2.raw_L2, a2.theta], lr=A2_COV_LR)
    best_state, best_eval = run_stage_with_checkpointing(
        a2,
        forecast_ensemble,
        opt1,
        A2_STAGE1_ITERS,
        A2_EVAL_EVERY,
        A2_EVAL_DRAWS,
        A2_EVAL_SEED,
        budget_limit,
        best_state,
        best_eval,
        "A2 geometry",
    )

    a2.load_state_dict(best_state)
    a2.to(DEVICE)

    # Stage 2: joint geometry + sensor logits.
    a2.logits.requires_grad_(True)
    opt2 = optim.Adam(
        [
            {"params": [a2.raw_L1, a2.raw_L2, a2.theta], "lr": A2_COV_LR},
            {"params": [a2.logits], "lr": A2_LOGITS_LR},
        ]
    )
    best_state, best_eval = run_stage_with_checkpointing(
        a2,
        forecast_ensemble,
        opt2,
        A2_STAGE2_ITERS,
        A2_EVAL_EVERY,
        A2_EVAL_DRAWS,
        A2_EVAL_SEED,
        budget_limit,
        best_state,
        best_eval,
        "A2 joint",
    )

    a2.load_state_dict(best_state)
    a2.to(DEVICE)
    return a2


def train_a3_from_a2(a2_model, forecast_ensemble, lat, lon):
    a3 = A3Model(lat, lon, DEVICE).to(DEVICE)
    a3.initialize_from_a2(a2_model)

    baseline = evaluate_forecast_fixed(
        a3, forecast_ensemble, seed=A3_EVAL_SEED, draws=A3_EVAL_DRAWS
    )
    budget_limit = max(BUDGET_OBS, baseline["expected_obs"])
    best_state = cpu_state_dict(a3)
    best_eval = dict(baseline)

    # Stage 1: NEW nonstationary warp only.
    a3.logits.requires_grad_(False)
    a3.raw_L1.requires_grad_(False)
    a3.raw_L2.requires_grad_(False)
    a3.theta.requires_grad_(False)
    for p in a3.warp.parameters():
        p.requires_grad_(True)

    opt1 = optim.Adam(a3.warp.parameters(), lr=A3_WARP_LR)
    best_state, best_eval = run_stage_with_checkpointing(
        a3,
        forecast_ensemble,
        opt1,
        A3_STAGE1_ITERS,
        A3_EVAL_EVERY,
        A3_EVAL_DRAWS,
        A3_EVAL_SEED,
        budget_limit,
        best_state,
        best_eval,
        "A3 warp only",
    )

    a3.load_state_dict(best_state)
    a3.to(DEVICE)

    # Stage 2: joint warp + A2 global geometry + sensor logits.
    a3.logits.requires_grad_(True)
    a3.raw_L1.requires_grad_(True)
    a3.raw_L2.requires_grad_(True)
    a3.theta.requires_grad_(True)
    for p in a3.warp.parameters():
        p.requires_grad_(True)

    opt2 = optim.Adam(
        [
            {"params": a3.warp.parameters(), "lr": A3_WARP_LR},
            {"params": [a3.raw_L1, a3.raw_L2, a3.theta], "lr": A3_COV_LR},
            {"params": [a3.logits], "lr": A3_LOGITS_LR},
        ]
    )
    best_state, best_eval = run_stage_with_checkpointing(
        a3,
        forecast_ensemble,
        opt2,
        A3_STAGE2_ITERS,
        A3_EVAL_EVERY,
        A3_EVAL_DRAWS,
        A3_EVAL_SEED,
        budget_limit,
        best_state,
        best_eval,
        "A3 joint",
    )

    a3.load_state_dict(best_state)
    a3.to(DEVICE)
    return a3


# =============================================================================
# Final CLEAN-truth evaluation
# =============================================================================


@torch.no_grad()
def evaluate_model_on_truth(model, clean_norm, clean_std_m, day_index):
    model.eval()
    phi_grid = model.geometry_embedding()
    rmse_values = []
    ev_values = []
    sensor_values = []

    truth_var = torch.var(clean_norm.float(), unbiased=True).clamp_min(1e-12)

    for draw in range(FINAL_EVAL_DRAWS):
        seed = FINAL_EVAL_SEED + day_index * 10000 + draw
        mask = hard_mask_from_fixed_seed(model.logits, seed)[0]
        selected = clean_norm * mask
        selected = selected.clone()
        selected[mask == 0] = float("nan")

        recon = optimal_interpolation_embedding(selected, phi_grid)
        err = recon - clean_norm
        rmse_norm = torch.sqrt(torch.mean(err.float() ** 2))
        rmse_values.append(float(rmse_norm.cpu()) * float(clean_std_m))

        ev = 1.0 - torch.var(err.float(), unbiased=True) / truth_var
        ev_values.append(float(ev.cpu()))
        sensor_values.append(float(mask.sum().cpu()))

    return {
        "RMSE (m)": float(np.mean(rmse_values)),
        "EV": float(np.mean(ev_values)),
        "Sensors": float(np.mean(sensor_values)),
    }


@torch.no_grad()
def evaluate_random_on_truth(clean_norm, clean_std_m, day_index, lat, lon):
    # One random result per day; it is forecast-error independent and is reused
    # for all 19 conditions of that day.
    yy, xx = torch.meshgrid(
        torch.arange(lat, dtype=torch.float32, device=DEVICE),
        torch.arange(lon, dtype=torch.float32, device=DEVICE),
        indexing="ij",
    )
    coords = torch.stack([xx, yy], dim=-1)

    # Match the validated Experiment-A random baseline exactly:
    # untrained Gumbel placement at p=0.001 + fixed isotropic L=25.
    phi_grid = coords / 25.0

    rmse_values = []
    ev_values = []
    sensor_values = []
    truth_var = torch.var(clean_norm.float(), unbiased=True).clamp_min(1e-12)

    for draw in range(FINAL_EVAL_DRAWS):
        seed = FINAL_EVAL_SEED + day_index * 10000 + draw
        random_eff = torch.full(
            (lat, lon),
            math.log(RATE / (1.0 - RATE)),
            dtype=torch.float32,
            device=DEVICE,
        )
        random_logits = torch.stack([torch.zeros_like(random_eff), random_eff], dim=0).unsqueeze(1)
        mask = hard_mask_from_fixed_seed(random_logits, seed)[0]

        if mask.sum() == 0:
            # Vanishingly unlikely; force one observation deterministically.
            mask.view(-1)[0] = 1.0

        selected = clean_norm * mask
        selected = selected.clone()
        selected[mask == 0] = float("nan")
        recon = optimal_interpolation_embedding(selected, phi_grid)

        err = recon - clean_norm
        rmse_norm = torch.sqrt(torch.mean(err.float() ** 2))
        rmse_values.append(float(rmse_norm.cpu()) * float(clean_std_m))

        ev = 1.0 - torch.var(err.float(), unbiased=True) / truth_var
        ev_values.append(float(ev.cpu()))
        sensor_values.append(float(mask.sum().cpu()))

    return {
        "RMSE (m)": float(np.mean(rmse_values)),
        "EV": float(np.mean(ev_values)),
        "Sensors": float(np.mean(sensor_values)),
    }


# =============================================================================
# Output / resume helpers
# =============================================================================


DAILY_COLUMNS = [
    "day_index",
    "date",
    "error_type",
    "severity",
    "forecast_RMSE_norm",
    "Method",
    "RMSE (m)",
    "EV",
    "Sensors",
]


def load_daily():
    if DAILY_CSV.exists():
        df = pd.read_csv(DAILY_CSV)
        missing = [c for c in DAILY_COLUMNS if c not in df.columns]
        if missing:
            raise RuntimeError(f"Existing {DAILY_CSV} is missing columns: {missing}")
        return df[DAILY_COLUMNS].copy()
    return pd.DataFrame(columns=DAILY_COLUMNS)


def state_key(day_index, error_type, severity):
    return f"day={int(day_index):02d}|error={error_type}|severity={severity}"


def new_state_bundle():
    return {
        "format_version": STATE_FORMAT_VERSION,
        "experiment": "Experiment B: 25-day forecast-error robustness/error attribution",
        "config": {
            "time_selection": "NATL60 time[::15], expected 25 days",
            "region": {"lat": [28.0, 45.0], "lon": [-66.0, -49.0], "border_cells": 30},
            "ensemble_size": ENSEMBLE_SIZE,
            "error_types": list(ERROR_TYPES),
            "severities": list(SEVERITIES),
            "deformation_factors": dict(DEFORMATION_FACTORS),
            "paper_displacement": {
                "L": PAPER_DISP_L,
                "T_corr": PAPER_DISP_TCORR,
                "sigma": PAPER_DISP_SIGMA,
                "seed_x": PAPER_DISP_SEED_X,
                "seed_y": PAPER_DISP_SEED_Y,
            },
            "perturb_seed": PERTURB_SEED,
            "model_seed": MODEL_SEED,
            "budget_obs": BUDGET_OBS,
            "sigma_f": SIGMA_F,
            "sigma_n": SIGMA_N,
            "A3": {
                "warp_architecture": "2-64-64-4 tanh",
                "warp_hidden": WARP_HIDDEN,
                "warp_dim": WARP_DIM,
                "alpha": ALPHA,
            },
            "final_eval_draws": FINAL_EVAL_DRAWS,
            "final_eval_seed": FINAL_EVAL_SEED,
        },
        "conditions": {},
    }


def load_states():
    if not STATES_PT.exists():
        return new_state_bundle()

    try:
        bundle = torch.load(STATES_PT, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(STATES_PT, map_location="cpu")

    if not isinstance(bundle, dict) or "conditions" not in bundle:
        raise RuntimeError(f"Existing {STATES_PT} is not a valid Experiment-B state bundle.")
    if int(bundle.get("format_version", -1)) != STATE_FORMAT_VERSION:
        raise RuntimeError(
            f"Unsupported state format in {STATES_PT}: "
            f"{bundle.get('format_version')} != {STATE_FORMAT_VERSION}"
        )
    return bundle


def export_condition_state(
    iso_model,
    a3_model,
    *,
    day_index,
    date_str,
    error_type,
    severity,
    mean_tgt,
    std_tgt,
    forecast_rmse,
    lat,
    lon,
):
    """Minimal state needed to reproduce masks/geometry and do later UQ."""
    with torch.no_grad():
        iso_eff = iso_model.effective_logit()[0].detach().cpu().float().clone()
        iso_L = float(iso_model.length_scale.detach().clamp_min(1e-3).cpu())

        a3_eff = a3_model.effective_logit()[0].detach().cpu().float().clone()
        a3_L1, a3_L2, a3_theta = a3_model.covariance_params()
        warp_state = {
            k: v.detach().cpu().float().clone()
            for k, v in a3_model.warp.state_dict().items()
        }

    return {
        "day_index": int(day_index),
        "date": str(date_str),
        "error_type": str(error_type),
        "severity": str(severity),
        "grid_shape": [int(lat), int(lon)],
        "normalization": {
            "clean_mean_m": float(mean_tgt),
            "clean_std_m": float(std_tgt),
        },
        "forecast_RMSE_norm": float(forecast_rmse),
        "reproducibility": {
            "day_perturb_seed_base": int(PERTURB_SEED + 1000 * int(day_index)),
            "iso_train_seed": int(condition_seed(day_index, error_type, severity, "iso")),
            "a2_train_seed": int(condition_seed(day_index, error_type, severity, "a2")),
            "a3_train_seed": int(condition_seed(day_index, error_type, severity, "a3")),
        },
        "Isotropic": {
            "effective_logit": iso_eff,
            "length_scale": iso_L,
        },
        "A3": {
            "effective_logit": a3_eff,
            "L1": float(a3_L1.detach().cpu()),
            "L2": float(a3_L2.detach().cpu()),
            "theta": float(a3_theta.detach().cpu()),
            "warp_state_dict": warp_state,
        },
    }


def condition_completed(df, state_bundle, day_index, error_type, severity):
    sub = df[
        (df["day_index"] == day_index)
        & (df["error_type"] == error_type)
        & (df["severity"] == severity)
    ]
    metrics_ok = {"A3", "Isotropic", "Random"}.issubset(set(sub["Method"].tolist()))
    state_ok = state_key(day_index, error_type, severity) in state_bundle.get("conditions", {})
    return metrics_ok and state_ok


def upsert_condition_rows(df, rows):
    row0 = rows[0]
    mask = (
        (df["day_index"] == row0["day_index"])
        & (df["error_type"] == row0["error_type"])
        & (df["severity"] == row0["severity"])
    )
    df = df.loc[~mask].copy()
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return df[DAILY_COLUMNS]


def write_summary(daily_df):
    if daily_df.empty:
        return

    summary = (
        daily_df.groupby(["error_type", "severity", "Method"], as_index=False)
        .agg(
            {
                "forecast_RMSE_norm": "mean",
                "RMSE (m)": "mean",
                "EV": "mean",
                "Sensors": "mean",
            }
        )
    )

    summary["_error_order"] = summary["error_type"].map(ERROR_ORDER)
    summary["_severity_order"] = summary["severity"].map(SEVERITY_ORDER)
    summary["_method_order"] = summary["Method"].map(METHOD_ORDER)
    summary = summary.sort_values(
        ["_error_order", "_severity_order", "_method_order"]
    ).drop(columns=["_error_order", "_severity_order", "_method_order"])

    # Strict compact summary: only the scenario labels and requested metrics.
    summary = summary[
        [
            "error_type",
            "severity",
            "Method",
            "forecast_RMSE_norm",
            "RMSE (m)",
            "EV",
            "Sensors",
        ]
    ]
    atomic_write_csv(summary, SUMMARY_CSV)


# =============================================================================
# Main experiment
# =============================================================================


def main():
    print(f"Device: {DEVICE}")
    print(f"Data:   {DATA_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(
        f"Experiment B: 25 days, {len(CONDITIONS)} conditions/day, "
        f"{ENSEMBLE_SIZE} forecast members/condition"
    )
    print("Final summary methods: A3, Isotropic, Random")
    print("Outputs: experiment_B_daily.csv + experiment_B_summary.csv + experiment_B_states.pt")

    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(DATA_PATH)

    tgt_ds = xr.open_dataset(DATA_PATH).isel(time=slice(0, None, 15))
    if tgt_ds.sizes.get("time") != 25:
        raise RuntimeError(
            f"Expected the same 25 selected days as Experiment A, got {tgt_ds.sizes.get('time')}."
        )

    start_day = max(0, START_DAY)
    end_day = min(25, END_DAY)
    if not (start_day < end_day):
        raise ValueError(f"Invalid day range [{start_day}, {end_day}).")

    # Determine full Gulf-Stream region shape once, then precompute the exact
    # paper displacement ensemble.  The paper code uses the same displacement
    # ensemble for every day (seeds 41/42), which we preserve.
    probe = tgt_ds.ssh.isel(time=0).sel(lat=slice(28, 45.0), lon=slice(-66, -49.0))
    full_H, full_W = probe.shape
    if full_H != full_W:
        raise RuntimeError(f"Paper displacement generator assumes square region, got {(full_H, full_W)}")

    print(
        f"Precomputing paper displacement fields: N={full_H}, L={PAPER_DISP_L}, "
        f"Tcorr={PAPER_DISP_TCORR}, sigma={PAPER_DISP_SIGMA}, J={ENSEMBLE_SIZE}"
    )
    base_dx = generate_correlated_fields(
        full_H,
        PAPER_DISP_L,
        PAPER_DISP_TCORR,
        PAPER_DISP_SIGMA,
        num_fields=ENSEMBLE_SIZE,
        device=DEVICE,
        seed=PAPER_DISP_SEED_X,
    )
    base_dy = generate_correlated_fields(
        full_H,
        PAPER_DISP_L,
        PAPER_DISP_TCORR,
        PAPER_DISP_SIGMA,
        num_fields=ENSEMBLE_SIZE,
        device=DEVICE,
        seed=PAPER_DISP_SEED_Y,
    )

    daily_df = load_daily()
    state_bundle = load_states()

    for day_index in range(start_day, end_day):
        time_step = tgt_ds.time.isel(time=day_index)
        date_str = str(time_step.values)[:10]
        print(f"\n================ DAY {day_index+1:02d}/25 | {date_str} ================", flush=True)

        inp_da = tgt_ds.ssh.sel(time=time_step)
        inp_full = inp_da.sel(lat=slice(28, 45.0), lon=slice(-66, -49.0)).fillna(0.0)
        border = 30
        inp_crop = inp_full.isel(
            lat=slice(border, inp_full.sizes["lat"] - border),
            lon=slice(border, inp_full.sizes["lon"] - border),
        ).fillna(0.0)

        lat, lon = inp_crop.shape
        if (lat, lon) != (280, 280):
            raise RuntimeError(f"Expected final 280x280 crop, got {(lat, lon)}")

        # IMPORTANT: normalize ONCE from the clean crop, then perturb.  We do
        # not renormalize each imperfect forecast, otherwise amplitude/bias
        # perturbations could disappear.
        mean_tgt = float(inp_crop.mean().item())
        std_tgt = float(inp_crop.std().item())
        if std_tgt <= 0:
            raise RuntimeError(f"Non-positive SSH std on day {day_index}: {std_tgt}")

        clean_full = torch.from_numpy(inp_full.values).float().to(DEVICE)
        clean_crop = torch.from_numpy(inp_crop.values).float().to(DEVICE)
        clean_full_norm = (clean_full - mean_tgt) / std_tgt
        clean_crop_norm = (clean_crop - mean_tgt) / std_tgt

        # Random is independent of forecast error: compute exactly once/day.
        random_metrics = evaluate_random_on_truth(
            clean_crop_norm, std_tgt, day_index, lat, lon
        )

        # Generate all forecast ensembles once/day, then release each condition
        # after use.  This keeps perturbation definitions perfectly paired.
        all_ensembles = build_all_forecast_ensembles(
            clean_full_norm,
            clean_crop_norm,
            border,
            day_index,
            base_dx,
            base_dy,
        )

        for condition_index, (error_type, severity) in enumerate(CONDITIONS):
            if condition_completed(
                daily_df, state_bundle, day_index, error_type, severity
            ):
                print(f"  [resume] skip {error_type}/{severity}", flush=True)
                continue

            forecast_ensemble = all_ensembles[(error_type, severity)]
            forecast_rmse = mean_ensemble_rmse(forecast_ensemble, clean_crop_norm)

            print(
                f"\n  --- {error_type.upper()} / {severity.upper()} "
                f"| forecast RMSE(norm)={forecast_rmse:.4f} ---",
                flush=True,
            )

            # -----------------------------------------------------------------
            # 1) Isotropic, trained only from the imperfect forecast ensemble.
            # -----------------------------------------------------------------
            set_all_seeds(condition_seed(day_index, error_type, severity, "iso"))
            iso = IsotropicModel(lat, lon, DEVICE).to(DEVICE)
            iso = train_isotropic(iso, forecast_ensemble)

            iso_metrics = evaluate_model_on_truth(
                iso, clean_crop_norm, std_tgt, day_index
            )
            print(
                f"    CLEAN truth eval Isotropic: RMSE={iso_metrics['RMSE (m)']:.6f} m | "
                f"EV={iso_metrics['EV']:.6f} | sensors={iso_metrics['Sensors']:.2f}",
                flush=True,
            )

            # -----------------------------------------------------------------
            # 2) A2 internal bridge.  No clean-truth model is ever loaded.
            # -----------------------------------------------------------------
            set_all_seeds(condition_seed(day_index, error_type, severity, "a2"))
            a2 = train_a2_from_isotropic(iso, forecast_ensemble, lat, lon)

            # -----------------------------------------------------------------
            # 3) A3 exact zero-warp warm start from THIS CONDITION'S A2.
            # -----------------------------------------------------------------
            set_all_seeds(condition_seed(day_index, error_type, severity, "a3"))
            a3 = train_a3_from_a2(a2, forecast_ensemble, lat, lon)

            a3_metrics = evaluate_model_on_truth(
                a3, clean_crop_norm, std_tgt, day_index
            )
            print(
                f"    CLEAN truth eval A3:       RMSE={a3_metrics['RMSE (m)']:.6f} m | "
                f"EV={a3_metrics['EV']:.6f} | sensors={a3_metrics['Sensors']:.2f}",
                flush=True,
            )
            print(
                f"    Random reference:          RMSE={random_metrics['RMSE (m)']:.6f} m | "
                f"EV={random_metrics['EV']:.6f} | sensors={random_metrics['Sensors']:.2f}",
                flush=True,
            )

            rows = []
            for method, metrics in [
                ("A3", a3_metrics),
                ("Isotropic", iso_metrics),
                ("Random", random_metrics),
            ]:
                rows.append(
                    {
                        "day_index": day_index,
                        "date": date_str,
                        "error_type": error_type,
                        "severity": severity,
                        "forecast_RMSE_norm": forecast_rmse,
                        "Method": method,
                        "RMSE (m)": metrics["RMSE (m)"],
                        "EV": metrics["EV"],
                        "Sensors": metrics["Sensors"],
                    }
                )

            # Save only the minimal consolidated model state needed for later
            # mask regeneration / posterior-covariance UQ.  No per-day files.
            key = state_key(day_index, error_type, severity)
            state_bundle["conditions"][key] = export_condition_state(
                iso,
                a3,
                day_index=day_index,
                date_str=date_str,
                error_type=error_type,
                severity=severity,
                mean_tgt=mean_tgt,
                std_tgt=std_tgt,
                forecast_rmse=forecast_rmse,
                lat=lat,
                lon=lon,
            )

            # Three cumulative files only.  Atomic writes make completed
            # condition boundaries safe for resume.
            daily_df = upsert_condition_rows(daily_df, rows)
            atomic_write_csv(daily_df, DAILY_CSV)
            write_summary(daily_df)
            atomic_torch_save(state_bundle, STATES_PT)

            del iso, a2, a3
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del all_ensembles, clean_full, clean_crop, clean_full_norm, clean_crop_norm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_summary(daily_df)
    atomic_torch_save(state_bundle, STATES_PT)
    print("\nDone.")
    print(f"Raw per-day metrics: {DAILY_CSV}")
    print(f"Compact summary:     {SUMMARY_CSV}")
    print(f"Consolidated states: {STATES_PT}")


if __name__ == "__main__":
    main()
