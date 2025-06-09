# data_denoising/sindy_chemotaxis_model.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary, IdentityLibrary
from pysindy.optimizers import STLSQ
from sklearn.preprocessing import StandardScaler

import subprocess

#  Setup for experiments
all_run_results = [] # Store results from all runs

# Define noise levels to process
noise_levels_to_process = ["00", "01", "05", "10", "25", "50"]

# Define STLSQ thresholds to test for NOISY data
experimental_thresholds_for_noisy = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]

# Fixed threshold for the zero-noise case
zero_noise_sindy_threshold = 1e-7

# True D and Chi values for plotting reference
D_true = 0.01
chi_true = 0.5

#  Looping over noise levels and SINDy thresholds

for level in noise_levels_to_process:
    fname = f"data_denoising/data/chemotaxis_{level}.npz"
    data  = np.load(fname)
    print(f"\nProcessing noise level: {float(level)/100:.2f}")

    U_noisy    = data["U_noisy"]
    U_t_noisy  = data["U_t_noisy"]
    U_x_noisy  = data["U_x_noisy"] 
    U_xx_noisy = data["U_xx_noisy"]
    U_ds_noisy = data["U_ds_noisy"]
    x_grid     = data["x"]
    t_grid     = data["t"]

    dt = t_grid[1] - t_grid[0]
    dx = x_grid[1] - x_grid[0]

    # --- PHASE 1---
    run_phase_1 = True 
    if run_phase_1:
        u_flat   = U_noisy.ravel()
        ut_flat  = U_t_noisy.ravel()
        ux_flat  = U_x_noisy.ravel()
        uxx_flat = U_xx_noisy.ravel()
        uds_flat = U_ds_noisy.ravel()
        Θ1        = np.vstack([u_flat, ux_flat, uxx_flat, uds_flat]).T
        Θ1_scaled = StandardScaler().fit_transform(Θ1)
        model1 = SINDy(
            feature_library=PolynomialLibrary(degree=2),
            optimizer=STLSQ(threshold=0.10)
        )
        model1.fit(Θ1_scaled, x_dot=ut_flat, t=dt)
        print(f" Phase 1 (degree-2) at noise = {float(level)/100:.2f}")
        model1.print()
    # else:
    #     print(f"Skipping Phase 1 for noise = {float(level)/100:.2f}")

    # --- PHASE 2: Chemotaxis PDE ---
    chem_flux = -np.gradient(U_ds_noisy, dx, axis=1)
    ut_vec   = U_t_noisy.ravel()
    uxx_vec  = U_xx_noisy.ravel()
    flux_vec = chem_flux.ravel()
    Θ2_raw = np.vstack([uxx_vec, flux_vec]).T

    coefs_ls, _, _, _ = np.linalg.lstsq(Θ2_raw, ut_vec, rcond=None)
    D_ls, chi_ls = coefs_ls[0], coefs_ls[1]
    # print(f"Direct LS (no scaling) → D ≃ {D_ls:.6f}, χ ≃ {chi_ls:.6f}")

    scaler2   = StandardScaler().fit(Θ2_raw)
    Θ2_scaled = scaler2.transform(Θ2_raw)

    thresholds_to_use_for_current_level = []
    if level == "00":
        thresholds_to_use_for_current_level = [zero_noise_sindy_threshold]
    else:
        thresholds_to_use_for_current_level = experimental_thresholds_for_noisy

    for current_threshold in thresholds_to_use_for_current_level:
        print(f"  Testing SINDy Phase 2 with threshold: {current_threshold:.1e}")
        model2 = SINDy(
            feature_library=IdentityLibrary(),
            optimizer=STLSQ(threshold=current_threshold)
        )
        model2.fit(Θ2_scaled, x_dot=ut_vec)

        coef_scaled = model2.coefficients().ravel()
        D_sindy, chi_sindy = 0.0, 0.0

        if len(coef_scaled) >= 1 and scaler2.scale_[0] != 0:
            D_sindy = coef_scaled[0] / scaler2.scale_[0]
        if len(coef_scaled) >= 2 and scaler2.scale_[1] != 0:
            chi_sindy = coef_scaled[1] / scaler2.scale_[1]
        
        if len(coef_scaled) < 2 :
             print(f"    Warning: SINDy returned {len(coef_scaled)} coefficients. Expected 2.")


        all_run_results.append({
            "noise_level_str": level, 
            "noise":     float(level)/100,
            "D_ls":      D_ls,
            "chi_ls":    chi_ls,
            "D_sindy":   D_sindy,
            "chi_sindy": chi_sindy,
            "sindy_threshold": current_threshold
        })


#  Display Results Table
df_results = pd.DataFrame(all_run_results)
print("\n----Full Results Table----")
# Define columns to print to ensure order and inclusion of new column
columns_to_print = ["noise", "sindy_threshold", "D_ls", "chi_ls", "D_sindy", "chi_sindy"]
print(df_results.to_string(index=False, columns=[col for col in columns_to_print if col in df_results.columns]))

print("\nGenerating plots.")

# Plotting D_sindy vs. SINDy Threshold for each noise level
plt.figure(figsize=(12, 7))
unique_noisy_levels = sorted(df_results[df_results["noise"] > 0]["noise_level_str"].unique())

for noise_str in unique_noisy_levels:
    subset = df_results[df_results["noise_level_str"] == noise_str]
    plt.plot(subset["sindy_threshold"], subset["D_sindy"], marker='o', linestyle='-', label=f'Noise {float(noise_str)/100:.2f}')

plt.axhline(D_true, color='r', linestyle='--', label=f'True D = {D_true}')
plt.xscale('log') 
plt.xlabel("SINDy Threshold (STLSQ)")
plt.ylabel("Discovered D_sindy")
plt.title("Discovered D vs. SINDy Threshold for Different Noise Levels")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.tight_layout()
plt.savefig("sindy_D_vs_threshold.png") 
plt.show()

# Plotting chi_sindy vs. SINDy Threshold for each noise level
plt.figure(figsize=(12, 7))
for noise_str in unique_noisy_levels:
    subset = df_results[df_results["noise_level_str"] == noise_str]
    plt.plot(subset["sindy_threshold"], subset["chi_sindy"], marker='o', linestyle='-', label=f'Noise {float(noise_str)/100:.2f}')

plt.axhline(chi_true, color='r', linestyle='--', label=f'True chi = {chi_true}')
plt.xscale('log') 
plt.xlabel("SINDy Threshold (STLSQ)")
plt.ylabel("Discovered chi_sindy")
plt.title("Discovered chi vs. SINDy Threshold for Different Noise Levels")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.tight_layout()
plt.savefig("sindy_chi_vs_threshold.png") # Save the plot
plt.show()

#  zero-noise case result 
zero_noise_result = df_results[df_results["noise"] == 0]
if not zero_noise_result.empty:
    print("\n--- Zero Noise Result (for reference) ---")
    print(zero_noise_result[["noise", "sindy_threshold", "D_sindy", "chi_sindy"]].to_string(index=False))

print("\nDone with SINDy analysis and plotting for sindy_chemotaxis_model.py ")

