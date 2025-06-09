# File: data_denoising/sindy_chemotaxis_model.py

import numpy as np
from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary, IdentityLibrary
from pysindy.optimizers import STLSQ
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
#  Load exactly the arrays that generate_chemotaxis_derivatives.py writes:
#    U_noisy    (nt-2 × nx-2) = noisy, normalized u(t,x)
#    U_t_noisy  (nt-2 × nx-2) = ∂ₜu computed by finite‐difference
#    U_x_noisy  (nt-2 × nx-2) = ∂ₓu
#    U_xx_noisy (nt-2 × nx-2) = ∂ₓₓu
#    U_ds_noisy (nt-2 × nx-2) = u·∂ₓS  (the chemotaxis interaction term)
#    x          (nx-2,)       = spatial grid (after cropping)
#    t          (nt-2,)       = time grid  (after cropping)
# ---------------------------------------------------------------------
data = np.load("data_denoising/data/chemotaxis_00.npz")

U_noisy   = data["U_noisy"]   # shape = (nt-2, nx-2)
U_t_noisy = data["U_t_noisy"] # same shape
U_x_noisy = data["U_x_noisy"]
U_xx_noisy= data["U_xx_noisy"]
U_ds_noisy= data["U_ds_noisy"] # this is u·∂ₓS
x_grid     = data["x"]        # length = nx-2
t_grid     = data["t"]        # length = nt-2

dt = t_grid[1] - t_grid[0]
dx = x_grid[1] - x_grid[0]

# -------------------------------------------------------
#  PHASE 1 (optional): run a generic degree‑2 SINDy on
#  [u, uₓ, uₓₓ, u·∂ₓS] → see which terms survive.
#  (If you only care about Phase 2, you can skip this.)
# -------------------------------------------------------
#  Flatten each 2D field into a long 1D vector of length N = (nt-2)*(nx-2)
u_flat   = U_noisy.ravel()
ut_flat  = U_t_noisy.ravel()
ux_flat  = U_x_noisy.ravel()
uxx_flat = U_xx_noisy.ravel()
uds_flat = U_ds_noisy.ravel()  # this is u·∂ₓS

#  Build Θ₁ = [u, uₓ, uₓₓ, u·∂ₓS] as (N × 4) and standardize
Θ1 = np.vstack([u_flat, ux_flat, uxx_flat, uds_flat]).T  # shape = (N, 4)
Θ1_scaled = StandardScaler().fit_transform(Θ1)

model1 = SINDy(
    feature_library = PolynomialLibrary(degree=2),
    optimizer       = STLSQ(threshold=0.10)
)
model1.fit(Θ1_scaled, x_dot=ut_flat, t=dt)

print("Phase 1: all degree 2 terms")
model1.print()

# --------------------------------------------------------
#  PHASE 2: Extract exactly two candidate columns for:
#
#    uₜ   =   D·uₓₓ   –   χ·∂ₓ( u·∂ₓS )
#
#  We already have:
#    U_xx_noisy   = uₓₓ
#    U_ds_noisy   = u·∂ₓS
#    U_t_noisy    = uₜ
#
#  We form a small “buffer = 1” so that ∂ₓ‐FD and ∂ₜ‐FD
#  never hit the boundary of U_noisy.  The generator script
#  has already trimmed one cell in each direction, so in
#  Phase 2 we only need a buffer=1 again to ensure no‐out‐of‐bounds.
# --------------------------------------------------------
buffer = 1

U_crop    = U_noisy   [buffer:-buffer, buffer:-buffer]    # shape = (nt-4, nx-4)
uxx_crop  = U_xx_noisy[buffer:-buffer, buffer:-buffer]
uds_crop  = U_ds_noisy[buffer:-buffer, buffer:-buffer]
ut_crop   = U_t_noisy [buffer:-buffer, buffer:-buffer]

t_crop = t_grid[buffer:-buffer]    # length = (nt-4)
x_crop = x_grid[buffer:-buffer]    # length = (nx-4)

print("\nAfter cropping for Phase 2:")
print("  U_crop   shape =", U_crop.shape)
print("  uₓₓ_crop shape =", uxx_crop.shape)
print("  (u·∂ₓS)_crop shape =", uds_crop.shape)
print("  uₜ_crop  shape =", ut_crop.shape)
print("  t_crop   length =", len(t_crop))
print("  x_crop   length =", len(x_crop))

# Flatten each cropped field into a long 1D array of length M = ( (nt-4)*(nx-4) )
uxx_vec      = uxx_crop.ravel()  
chem_flux_vec= (-np.gradient(uds_crop, dx, axis=1)).ravel()  # –∂ₓ( u·∂ₓS )
ut_vec       = ut_crop.ravel()  # this is uₜ, also flattened

# Build raw Θ₂ = [uₓₓ,  –∂ₓ(u·∂ₓS)] as (M × 2)
Θ2_raw = np.vstack([uxx_vec, chem_flux_vec]).T  # shape = (M, 2)

# ----------------------------------------------------------------------
#  OPTION A (direct least‐squares, no scaling):
#      [D, χ] = (Θ₂ᵀΘ₂)⁻¹ Θ₂ᵀ (uₜ)
#  Uncomment the lines below if you want a raw‐LS solution:
# ----------------------------------------------------------------------
coef_ls = np.linalg.lstsq(Θ2_raw, ut_vec, rcond=None)[0]
D_ls   = coef_ls[0]
chi_ls = coef_ls[1]
print(f"\nDirect LS →   D≃{D_ls:.6f},   χ≃{chi_ls:.6f}")

# ----------------------------------------------------------------------
#  OPTION B (scaled + IdentityLibrary + STLSQ threshold):
#  1) zero‑mean, unit‑variance each column of Θ₂_raw
#  2) fit with STLSQ( threshold=… )
#  3) un‑scale the two coefficients back to physical units
# ----------------------------------------------------------------------
scaler2  = StandardScaler()
Θ2_scaled = scaler2.fit_transform(Θ2_raw)

model2 = SINDy(
    feature_library = IdentityLibrary(),
    optimizer       = STLSQ(threshold=1e-7)
)
model2.fit(Θ2_scaled, x_dot=ut_vec, t=dt)

coef_scaled = model2.coefficients().ravel()  # length=2
scales      = scaler2.scale_                     # length=2

D_sindy   = coef_scaled[0] / scales[0]
chi_sindy = coef_scaled[1] / scales[1]

print("\n Phase 2 (zero-mean/unit-var Θ₂, threshold=1e-4): chemotaxis PDE")
print(f"Recovered D_sindy   = {D_sindy:.12f}")
print(f"Recovered chi_sindy = {chi_sindy:.12f}")


# If you also want to report the raw, un‑scaled‐LS solution, uncomment above.
