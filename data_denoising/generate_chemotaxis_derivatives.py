# data_denoising/generate_chemotaxis_derivatives.py

import numpy as np
import os

def generate_chemotaxis_data_with_derivatives(
    filename="data_denoising/data/chemotaxis_00.npz",
    D=0.01, chi=0.5, L=1.0, T=1.0, nx=199, nt=2000, sigma=0.0 # Increased nt for stability
):
    """
    1) Simulating u(t,x) with chemotaxis PDE: u_{t} = D u_{xx} - chi * ∂x(u * S_x).
       Uses a stable dt due to increased nt.
    2) For sigma=0, define u_t for SINDy using the true D, chi and the exact
       FD forms SINDy will use for its library. For sigma > 0, compute u_t via FD.
    3) Add proportional noise to u itself, but compute all derivatives on
       the *noisy* u. (Except u_t for sigma=0 as per point 2).
    4) Save arrays: U_noisy, U_t_noisy, U_x_noisy, U_xx_noisy, U_ds_noisy, x, t.
    """

    dx = L / (nx - 1)
    dt = T / (nt - 1) # dt is now smaller

    print(f"Generating data for {filename} with D={D}, chi={chi}, sigma={sigma}")
    print(f"Spatial step dx = {dx:.2e}, Temporal step dt = {dt:.2e}")
    print(f"Stability check (D*dt/dx^2): {D*dt/(dx**2):.4f} (should be < 0.5 for simple diffusion)")


    # full spatial grid, length = nx
    x = np.linspace(0, L, nx)  
     # full time grid, length = nt  
    t = np.linspace(0, T, nt)   

    # Simulate the PDE on the *raw* (u) grid:
    u_simulated = np.zeros((nt, nx)) 
    u_simulated[0, :] = 0.1 + 0.05 * np.exp(-((x - 0.3) ** 2) / 0.01)

    S    = np.exp(-((x - 0.7) ** 2) / 0.02)
    dSdx = np.gradient(S, dx) # dS/dx on the full x-grid

    for n in range(nt - 1):
        dudx_sim      = np.gradient(u_simulated[n, :], dx)
        d2udx2_sim    = np.gradient(dudx_sim, dx)
        chem_flux_sim = np.gradient(u_simulated[n, :] * dSdx, dx)
        
        u_simulated[n + 1, :] = u_simulated[n, :] + dt * (D * d2udx2_sim - chi * chem_flux_sim)

    print(f"Simulation finished. Max u_simulated value: {np.max(u_simulated):.4f}, Min u_simulated value: {np.min(u_simulated):.4f}")

    #  Add proportional Gaussian noise to *u_simulated* itself:
    if sigma > 0:
        noise = sigma * u_simulated * np.random.randn(*u_simulated.shape)
        u_for_derivatives = u_simulated + noise 
    else:
        u_for_derivatives = u_simulated 

    # Compute library terms for SINDy from u_for_derivatives
    U_xx_candidate = np.gradient(np.gradient(u_for_derivatives, dx, axis=1), dx, axis=1)
    U_ds_candidate = u_for_derivatives * dSdx
    Partial_x_U_ds_candidate = np.gradient(U_ds_candidate, dx, axis=1)

    # Define the target U_t for SINDy.
    if sigma == 0:
        U_t_defined_for_sindy = D * U_xx_candidate - chi * Partial_x_U_ds_candidate
    else:
        U_t_defined_for_sindy = np.gradient(u_for_derivatives, dt, axis=0)
    # For saving U_x_noisy
    U_x_for_sindy  = np.gradient(u_for_derivatives, dx, axis=1) 

    # Crop boundaries to reduce dimensions from (nt, nx) to (nt-2, nx-2)
    u_final_crop    = u_for_derivatives[1:-1, 1:-1]
    u_t_final_crop  = U_t_defined_for_sindy[1:-1, 1:-1] 
    u_x_final_crop  = U_x_for_sindy[1:-1, 1:-1]
    u_xx_final_crop = U_xx_candidate[1:-1, 1:-1] 
    uds_final_crop  = U_ds_candidate[1:-1, 1:-1] 

    # Corresponds to nx-2 points
    x_crop = x[1:-1] 
    # Corresponds to nt-2 points
    t_crop = t[1:-1] 

    #  Save these six arrays in one .npz:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez(
        filename,
        U_noisy    = u_final_crop,       
        U_t_noisy  = u_t_final_crop,     
        U_x_noisy  = u_x_final_crop,     
        U_xx_noisy = u_xx_final_crop,    
        U_ds_noisy = uds_final_crop,     
        x          = x_crop,
        t          = t_crop
    )
    print(f"Saved to “{filename}” with U_noisy.shape = {u_final_crop.shape} (nt_cropped, nx_cropped)")
    print(f"  Max U_noisy value in saved file: {np.max(u_final_crop):.4f}, Min: {np.min(u_final_crop):.4f}")


if __name__ == "__main__":
    true_D = 0.01
    true_chi = 0.5
    
    # Set nt for the simulation 
    simulation_nt = 2000 # Using the new stable value

    noise_levels = ["00","01","05","10","25","50"]
    
    
    for level in noise_levels:
        sigma_val = float("0." + level) if level != "00" else 0.0
        fn    = f"data_denoising/data/chemotaxis_{level}.npz" 
        generate_chemotaxis_data_with_derivatives(
            filename=fn,
            D=true_D,      
            chi=true_chi,  
            nt=simulation_nt, # Pass the new nt
            sigma=sigma_val
        )
        print("-" * 50)