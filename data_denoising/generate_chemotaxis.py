import numpy as np
import os
import matplotlib.pyplot as plt 

def generate_chemotaxis_data(filename="data_denoising/data/chemotaxis_data.npz", 
                              D=0.01, chi=0.5, L=1.0, T=1.0, nx=199, nt=99, sigma=0.1):
    """
    Simulate a chemotaxis PDE model and save noisy spatiotemporal data.

    PDE: du/dt = D * d2u/dx2 - chi * d/dx (u * dS/dx)
    """
    dx = L / (nx - 1)
    dt = T / (nt - 1)

    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)

    # Initial condition: localized bump
    u = np.zeros((nt, nx))
    u[0, :] = 0.1 + 0.05 * np.exp(-((x - 0.3) ** 2) / 0.01)

    # Static chemical signal S(x): Gaussian centered at 0.7
    S = np.exp(-((x - 0.7) ** 2) / 0.02)
    dSdx = np.gradient(S, dx)

    # Time stepping using explicit finite differences
    for n in range(0, nt - 1):
        dudx = np.gradient(u[n, :], dx)
        d2udx2 = np.gradient(dudx, dx)
        chem_flux = np.gradient(u[n, :] * dSdx, dx)
        u[n + 1, :] = u[n, :] + dt * (D * d2udx2 - chi * chem_flux)

    # Normalize to [0, 1]
    u_min, u_max = np.min(u), np.max(u)
    u = (u - u_min) / (u_max - u_min)

    # Add proportional Gaussian noise
    noise = sigma * u * np.random.randn(*u.shape)
    u_noisy = u + noise

    # Save as .npz file
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez(filename, u=u_noisy, x=x, t=t)

    print(f"Chemotaxis data saved to {filename} with shape {u_noisy.shape}")

if __name__ == "__main__":
    noise_levels = ['00', '01', '05', '10', '25', '50']
    for level in noise_levels:
        sigma = float('0.' + level) if level != '00' else 0.0
        filename = f"data_denoising/data/chemotaxis_{level}.npz"
        generate_chemotaxis_data(filename=filename, sigma=sigma)

data_00 = np.load("data_denoising/data/chemotaxis_00.npz")['u']
data_50 = np.load("data_denoising/data/chemotaxis_50.npz")['u']

plt.plot(data_00[-1], label='No noise (00)')
plt.plot(data_50[-1], label='High noise (50)')
plt.legend()
plt.title("Final time step comparison")
plt.xlabel("x index")
plt.ylabel("u")
plt.show()
for level in ['00', '01', '05', '10', '25', '50']:
    path = f"data_denoising/data/chemotaxis_{level}.npz"
    data = np.load(path)['u']
    print(f"chemotaxis_{level}.npz shape: {data.shape}, total points: {data.size}")
