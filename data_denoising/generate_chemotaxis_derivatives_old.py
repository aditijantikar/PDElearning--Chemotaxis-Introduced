import numpy as np
import os

def generate_chemotaxis_data_with_derivatives(
    filename="data_denoising/data/chemotaxis_00.npz",
    D=0.01, chi=0.5, L=1.0, T=1.0, nx=199, nt=99, sigma=0.1
):
    dx = L / (nx - 1)
    dt = T / (nt - 1)

    # 1) build grid and initial condition
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    u = np.zeros((nt, nx))
    u[0, :] = 0.1 + 0.05 * np.exp(-((x - 0.3) ** 2) / 0.01)

    # 2) static signal S(x) and its gradient
    S = np.exp(-((x - 0.7) ** 2) / 0.02)
    dSdx = np.gradient(S, dx)

    # 3) march forward in time (clean PDE)
    for n in range(nt - 1):
        dudx     = np.gradient(u[n, :], dx)
        d2udx2   = np.gradient(dudx, dx)
        chemflux = np.gradient(u[n, :] * dSdx, dx)
        u[n + 1, :] = u[n, :] + dt * (D * d2udx2 - chi * chemflux)

    # 4) normalize to [0,1] THEN add *proportional* Gaussian noise
    u_min, u_max = np.min(u), np.max(u)
    u = (u - u_min) / (u_max - u_min)
    noise = sigma * u * np.random.randn(*u.shape)
    u_noisy = u + noise

    # 5) compute all derivatives on the *noisy* field
    u_t   = np.gradient(u_noisy, dt, axis=0)
    u_x   = np.gradient(u_noisy, dx, axis=1)
    u_xx  = np.gradient(u_x, dx, axis=1)

    # 6) now compute u·∂ₓS on the *noisy* u
    u_times_dSdx = u_noisy * dSdx   # shape = (nt, nx)

    # 7) trim one‐cell buffer so no FD indexing goes out of bounds
    #    (we remove index 0 and index (−1) in both t‐ and x‐directions)
    u_crop     = u_noisy   [1:-1, 1:-1]
    u_t_crop   = u_t       [1:-1, 1:-1]
    u_x_crop   = u_x       [1:-1, 1:-1]
    u_xx_crop  = u_xx      [1:-1, 1:-1]
    u_ds_crop  = u_times_dSdx[1:-1, 1:-1]
    x_crop     = x[1:-1]
    t_crop     = t[1:-1]

    # 8) save exactly these six arrays, using the keys below:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez(
        filename,
        U_noisy    = u_crop,     # shape = (nt−2, nx−2)
        U_t_noisy  = u_t_crop,
        U_x_noisy  = u_x_crop,
        U_xx_noisy = u_xx_crop,
        U_ds_noisy = u_ds_crop,
        x          = x_crop,     # length = nx−2
        t          = t_crop      # length = nt−2
    )

    print(f"[generate] saved “{filename}” with U_noisy.shape = {u_crop.shape}")


if __name__ == "__main__":
    noise_levels = ['00','01','05','10','25','50']
    for level in noise_levels:
        sigma = float('0.' + level) if level != '00' else 0.0
        fname = f"data_denoising/data/chemotaxis_{level}.npz"
        generate_chemotaxis_data_with_derivatives(
            filename=fname,
            D=0.01, chi=0.5, L=1.0, T=1.0, nx=199, nt=99, sigma=sigma
        )