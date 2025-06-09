
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd


from pysindy import SINDy
from pysindy.feature_library import IdentityLibrary
from pysindy.optimizers import STLSQ
from sklearn.preprocessing import StandardScaler

# --- Chemotaxis Gradient Function ---
def dS_dx(x_np):
    return -((x_np - 0.7) / 0.01) * np.exp(-((x_np - 0.7)**2) / 0.02)

# --- Define Denoiser ANN with Skip Connections, tanh, Dropout ---
class DenoiserANN(nn.Module):
    def __init__(self, dropout_p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(p=dropout_p)
        self.out = nn.Linear(128, 1)

    def forward(self, x, t):
        xt = torch.stack([x, t], dim=-1)
        h1 = torch.tanh(self.fc1(xt))
        h2 = torch.tanh(self.fc2(h1))
        h2 = self.dropout(h2)
        return self.out(h1 + h2).squeeze(-1)  # Skip connection

# --- Train with Sobolev Regularization and Optional PDE Constraint ---
def train_ann(model, x, t, u, epochs=500, lr=1e-3, λ=1e-4, use_pde_constraint=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    D_nom, chi_nom = 0.01, 0.5

    for epoch in range(epochs):
        optimizer.zero_grad()
        x.requires_grad_(True)
        t.requires_grad_(True)

        u_pred = model(x, t)
        loss_mse = torch.mean((u_pred - u) ** 2)

        ux = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        uxx = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=True)[0]
        loss_smooth1 = torch.mean(ux ** 2)
        loss_smooth2 = torch.mean(uxx ** 2)

        loss = loss_mse + λ * (loss_smooth1 + loss_smooth2)

        if use_pde_constraint:
            ut = torch.autograd.grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
            dS = torch.tensor(dS_dx(x.detach().numpy()), dtype=torch.float32)
            u_dS = u_pred * dS
            dx_u_dS = torch.autograd.grad(u_dS, x, grad_outputs=torch.ones_like(u_dS), create_graph=True)[0]
            pde_residual = ut - D_nom * uxx + chi_nom * dx_u_dS
            loss_pde = torch.mean(pde_residual ** 2)
            loss += 0.1 * loss_pde  # Weight for PDE loss

        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"[Epoch {epoch}] MSE: {loss_mse.item():.3e}, Smooth: {loss_smooth2.item():.3e}")

    return model

# --- Plotting ANN vs. noisy vs. clean ---
def plot_u_comparison(model, x_grid, t_grid, U_noisy, U_clean):
    model.train()  # Keep dropout on
    X, T = np.meshgrid(x_grid, t_grid, indexing='ij')
    xt = torch.tensor(np.stack([X.ravel(), T.ravel()], axis=-1), dtype=torch.float32)

    with torch.no_grad():
        samples = [model(xt[:, 0], xt[:, 1]).cpu().numpy().reshape(X.shape) for _ in range(20)]
        U_ann_mean = np.mean(samples, axis=0)
        U_ann_std = np.std(samples, axis=0)

    plt.figure(figsize=(16, 4))
    titles = ['Noisy $u(x,t)$', 'Denoised ANN $u_{ANN}(x,t)$', 'Clean $u_{clean}(x,t)$']
    fields = [U_noisy, U_ann_mean, U_clean]

    for i, field in enumerate(fields):
        plt.subplot(1, 3, i + 1)
        plt.imshow(field, extent=[t_grid[0], t_grid[-1], x_grid[0], x_grid[-1]],
                   origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(titles[i])
        plt.xlabel("t")
        plt.ylabel("x")

    plt.tight_layout()
    plt.show()

# --- Run full pipeline for one dataset ---
def run_pipeline(filename, D_true=0.01, chi_true=0.5):
    data = np.load(filename)
    U, x, t = data['U_noisy'], data['x'], data['t']
    dt = t[1] - t[0]

    crop = 10
    U = U[crop:-crop, :]
    x = x[crop:-crop]

    X, T = np.meshgrid(x, t, indexing='ij')
    x_flat = torch.tensor(X.ravel(), dtype=torch.float32)
    t_flat = torch.tensor(T.ravel(), dtype=torch.float32)
    u_noisy = torch.tensor(U.ravel(), dtype=torch.float32)

    n = 10000
    indices = torch.randperm(x_flat.shape[0])[:n]
    x_flat = x_flat[indices]
    t_flat = t_flat[indices]
    u_noisy = u_noisy[indices]

    model = DenoiserANN(dropout_p=0.1)
    model = train_ann(model, x_flat, t_flat, u_noisy)
    model.eval()

    x_flat.requires_grad_(True)
    t_flat.requires_grad_(True)

    u_hat = model(x_flat, t_flat)
    ut = torch.autograd.grad(u_hat, t_flat, grad_outputs=torch.ones_like(u_hat), create_graph=True)[0]
    ux = torch.autograd.grad(u_hat, x_flat, grad_outputs=torch.ones_like(u_hat), create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x_flat, grad_outputs=torch.ones_like(ux), create_graph=True)[0]

    dS = torch.tensor(dS_dx(x_flat.detach().numpy()), dtype=torch.float32)
    u_dS = u_hat * dS
    dx_u_dS = torch.autograd.grad(u_dS, x_flat, grad_outputs=torch.ones_like(u_dS), create_graph=True)[0]

    Theta = torch.stack([uxx, -dx_u_dS], dim=1).detach().numpy()
    Ut = ut.detach().numpy().reshape(-1, 1)

    scaler = StandardScaler()
    Theta_scaled = scaler.fit_transform(Theta)

    sindy = SINDy(feature_library=IdentityLibrary(), optimizer=STLSQ(threshold=1e-3))
    sindy.fit(Theta_scaled, x_dot=Ut, t=dt)
    coeffs = sindy.coefficients().flatten()

    stds = scaler.scale_
    D_est, chi_est = coeffs[0] / stds[0], coeffs[1] / stds[1]

    err_D = abs(D_est - D_true) / D_true * 100
    err_chi = abs(chi_est - chi_true) / chi_true * 100

    return D_est, chi_est, err_D, err_chi, sindy, model, x, t, U, data['U_clean']

# --- Run Across Noise Levels ---
sigmas = ['00', '01', '05', '10', '25', '50']
results = []

for s in sigmas:
    print(f"\n--- Processing σ = 0.{s} ---")
    file_path = f"/content/chemotaxis_{s}.npz"
    try:
        D, chi, err_D, err_chi, sindy_model, ann_model, x, t, U_noisy, U_clean = run_pipeline(file_path)
        print("Recovered PDE:")
        sindy_model.print()
        results.append((f"0.{s}", D, chi, err_D, err_chi))

        if s == '10':
            plot_u_comparison(ann_model, x, t, U_noisy, U_clean)

    except Exception as e:
        print(f"Error processing σ = 0.{s}: {e}")

# --- Tabulate and Plot Results ---
df = pd.DataFrame(results, columns=['σ', 'D_ANN', 'χ_ANN', 'ε_D (%)', 'ε_χ (%)'])
print("\n==== ANN-SINDy Results Across Noise Levels ====")
print(df)

df['σ_float'] = df['σ'].astype(float)
plt.plot(df['σ_float'], df['ε_D (%)'], label='D error')
plt.plot(df['σ_float'], df['ε_χ (%)'], label='χ error')
plt.xlabel("Noise level σ")
plt.ylabel("Relative Error (%)")
plt.title("ANN-SINDy vs. Noise")
plt.legend()
plt.grid(True)
plt.show()