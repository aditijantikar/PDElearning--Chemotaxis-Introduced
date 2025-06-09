import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy.interpolate import griddata

# --- Configuration ---
D_val = 0.01
chi_val = 0.5
L_domain = 1.0
T_domain = 1.0

nn_hyperparams = {
    'num_hidden_layers': 5,
    'neurons_per_layer': 40
}

learning_rate = 1e-3
num_epochs = 30000

N_collocation = 20000
N_initial = 500
N_boundary = 500

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Helper Functions ---
def S_profile(x_tensor):
    return torch.exp(-((x_tensor - 0.7)**2) / 0.02)

def dSdx_profile(x_tensor):
    return -(x_tensor - 0.7) / 0.01 * torch.exp(-((x_tensor - 0.7)**2) / 0.02)

def initial_condition(x_tensor):
    return 0.1 + 0.05 * torch.exp(-((x_tensor - 0.3)**2) / 0.01)

# --- Neural Network ---
class PINN_Net(nn.Module):
    def __init__(self, num_hidden_layers=4, neurons_per_layer=50, input_dim=2, output_dim=1):
        super(PINN_Net, self).__init__()
        activation = nn.SiLU()
        layers = [nn.Linear(input_dim, neurons_per_layer), activation]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(activation)
        layers.append(nn.Linear(neurons_per_layer, output_dim))
        self.model = nn.Sequential(*layers)
        self.init_weights()

    def forward(self, x_t_input):
        return self.model(x_t_input)

    def init_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

# --- Chemotaxis PINN ---
class ChemotaxisPINN(nn.Module):
    def __init__(self, L_domain, T_domain, nn_config):
        super(ChemotaxisPINN, self).__init__()
        self.D = nn.Parameter(torch.tensor(0.01, dtype=torch.float32, device=device))
        self.chi = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=device))

        self.L = L_domain
        self.T = T_domain

        self.u_net = PINN_Net(
            num_hidden_layers=nn_config['num_hidden_layers'],
            neurons_per_layer=nn_config['neurons_per_layer']
        ).to(device)

        self.S_func = S_profile
        self.dSdx_func = dSdx_profile
        self.ic_func = initial_condition

        self.loss_history = []
        self.loss_pde_history = []
        self.loss_ic_history = []
        self.loss_bc_history = []

    def compute_pde_residual(self, x, t):
        u_val = self.u_net(torch.cat([x, t], dim=1))
        u_t = torch.autograd.grad(u_val, t, grad_outputs=torch.ones_like(u_val), create_graph=True)[0]
        u_x = torch.autograd.grad(u_val, x, grad_outputs=torch.ones_like(u_val), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        dSdx_val = self.dSdx_func(x)
        V = u_val * dSdx_val
        dVdx = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        return u_t - self.D * u_xx + self.chi * dVdx

    def compute_ic_loss(self, x_ic, t_ic):
        u_pred_ic = self.u_net(torch.cat([x_ic, t_ic], dim=1))
        u_true_ic = self.ic_func(x_ic)
        return torch.mean((u_pred_ic - u_true_ic)**2)

    def compute_bc_loss(self, x_bc_left, x_bc_right, t_bc):
        u_bc_left = self.u_net(torch.cat([x_bc_left, t_bc], dim=1))
        du_dx_left = torch.autograd.grad(u_bc_left, x_bc_left, grad_outputs=torch.ones_like(u_bc_left), create_graph=True)[0]
        loss_bc_left = torch.mean(du_dx_left**2)

        u_bc_right = self.u_net(torch.cat([x_bc_right, t_bc], dim=1))
        du_dx_right = torch.autograd.grad(u_bc_right, x_bc_right, grad_outputs=torch.ones_like(u_bc_right), create_graph=True)[0]
        loss_bc_right = torch.mean(du_dx_right**2)
        return loss_bc_left + loss_bc_right

    def total_loss(self, x_col, t_col, x_ic, t_ic, x_bc_left, x_bc_right, t_bc, weights=None):
        if weights is None:
            weights = {'pde': 1.0, 'ic': 1.0, 'bc': 1.0}

        loss_pde = torch.mean(self.compute_pde_residual(x_col, t_col)**2)
        loss_ic = self.compute_ic_loss(x_ic, t_ic)
        loss_bc = self.compute_bc_loss(x_bc_left, x_bc_right, t_bc)

        total = weights['pde'] * loss_pde + weights['ic'] * loss_ic + weights['bc'] * loss_bc

        if self.training:
            self.loss_history.append(total.item())
            self.loss_pde_history.append(loss_pde.item())
            self.loss_ic_history.append(loss_ic.item())
            self.loss_bc_history.append(loss_bc.item())
        return total

x_col = (torch.rand(N_collocation, 1, device=device) * L_domain).requires_grad_(True)
t_col = (torch.rand(N_collocation, 1, device=device) * T_domain).requires_grad_(True)
x_ic = torch.linspace(0, L_domain, N_initial, device=device).view(-1, 1)
t_ic = torch.zeros_like(x_ic)
t_bc = torch.linspace(0, T_domain, N_boundary, device=device).view(-1, 1)
x_bc_left = torch.zeros_like(t_bc).requires_grad_(True)
x_bc_right = torch.ones_like(t_bc).requires_grad_(True)

pinn_model = ChemotaxisPINN(L_domain, T_domain, nn_hyperparams)
optimizer = torch.optim.AdamW(
    list(pinn_model.u_net.parameters()) + [pinn_model.D, pinn_model.chi],
    lr=learning_rate
)

loss_weights = {'pde': 100.0, 'ic': 100.0, 'bc': 10.0}
print(f"Using loss weights: {loss_weights}")
print(f"Starting training for {num_epochs} epochs...")
start_time = time.time()
pinn_model.train()

for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = pinn_model.total_loss(x_col, t_col, x_ic, t_ic, x_bc_left, x_bc_right, t_bc, weights=loss_weights)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Total Weighted Loss: {loss.item():.4e}, "
              f"PDE Loss (unweighted): {pinn_model.loss_pde_history[-1]:.4e}, "
              f"IC Loss (unweighted): {pinn_model.loss_ic_history[-1]:.4e}, "
              f"BC Loss (unweighted): {pinn_model.loss_bc_history[-1]:.4e}, "
              f"D: {pinn_model.D.item():.5f}, chi: {pinn_model.chi.item():.5f}")

end_time = time.time()
print(f"Training finished in {end_time - start_time:.2f} seconds.")

# --- Plotting Loss History ---
plt.figure(figsize=(10, 6))
plt.plot(pinn_model.loss_history, label='Total Loss')
plt.plot(pinn_model.loss_pde_history, label='PDE Loss', alpha=0.7)
plt.plot(pinn_model.loss_ic_history, label='IC Loss', alpha=0.7)
plt.plot(pinn_model.loss_bc_history, label='BC Loss', alpha=0.7)
plt.xlabel(f'Epoch (logged every {1 if num_epochs < 1000 else len(pinn_model.loss_history)//(num_epochs//1000)} steps)')
plt.ylabel('Loss Value')
plt.yscale('log')
plt.title('Training Loss History')
plt.legend()
plt.grid(True)
plt.savefig("pinn_forward_loss_history.png")
plt.show()


# After training, we compare the PINN solution u_NN(x,t) with the finite difference solution (clean).

# Grid for plotting PINN solution
nx_plot = 197 # To match nx_cropped from FD data
nt_plot = 1998 # To match nt_cropped from FD data

# Create 1D tensors for x and t for plotting
x_plot_1d_torch = torch.linspace(0, L_domain, nx_plot, device=device)
t_plot_1d_torch = torch.linspace(0, T_domain, nt_plot, device=device)

# Create meshgrid for creating the input to the NN (for a full field prediction)
X_plot_mesh, T_plot_mesh = torch.meshgrid(x_plot_1d_torch, t_plot_1d_torch, indexing='xy')
x_flat_plot = X_plot_mesh.flatten().view(-1,1)
t_flat_plot = T_plot_mesh.flatten().view(-1,1)

# Get PINN predictions
pinn_model.u_net.eval() # Set model to evaluation mode
with torch.no_grad():
    u_pinn_solution_flat = pinn_model.u_net(torch.cat([x_flat_plot, t_flat_plot], dim=1))

# Reshape PINN solution to (nt_plot, nx_plot) for plotting with t on y-axis, x on x-axis
# view(nx_plot, nt_plot) would make it u(x,t) if x is first dim of meshgrid then .T gives u(t,x)

u_pinn_solution_grid = u_pinn_solution_flat.view(nx_plot, nt_plot).cpu().numpy() # Shape (nx_plot, nt_plot) -> u(x,t)
u_pinn_for_plot = u_pinn_solution_grid.T # Shape (nt_plot, nx_plot) -> u(t,x)

# Convert 1D plot grids to numpy for matplotlib
x_plot_1d_np = x_plot_1d_torch.cpu().numpy()
t_plot_1d_np = t_plot_1d_torch.cpu().numpy()

plt.figure(figsize=(10,6))
# For contourf(X,Y,Z), if X and Y are 1D, Z must be (len(Y), len(X))
plt.contourf(x_plot_1d_np, t_plot_1d_np, u_pinn_for_plot, levels=50, cmap='viridis')
plt.colorbar(label='u(x,t) PINN solution')
plt.xlabel('Space x')
plt.ylabel('Time t')
plt.title('PINN Solution for Chemotaxis PDE (Forward Problem)')
plt.savefig("pinn_forward_solution.png")
plt.show()

# START: Load Finite Difference (FD) solution and Compare

# Since this has been computed on Google Colab, the file path here is temporary. To run it locally, change the path name 
fd_data_path = "/content/chemotaxis_00.npz" 
if os.path.exists(fd_data_path):
    print(f"\nLoading Finite Difference solution from: {fd_data_path}")
    data_fd = np.load(fd_data_path)
    u_fd_clean_cropped = data_fd['U_noisy'] #clean solution (nt_cropped, nx_cropped)
    x_fd_cropped_1d = data_fd['x']         #1D x_crop
    t_fd_cropped_1d = data_fd['t']         # 1D t_crop

    print(f"FD solution shape: {u_fd_clean_cropped.shape}")
    print(f"FD x_crop shape: {x_fd_cropped_1d.shape}")
    print(f"FD t_crop shape: {t_fd_cropped_1d.shape}")

    # Points where FD data is known (original points for griddata)
    X_fd_mesh_orig, T_fd_mesh_orig = np.meshgrid(x_fd_cropped_1d, t_fd_cropped_1d)
    points_fd = np.vstack((X_fd_mesh_orig.ravel(), T_fd_mesh_orig.ravel())).T
    # griddata expects (N, D) for points and (N,) for values. u_fd is (t,x) so .T gives (x,t) then ravel
    values_fd = u_fd_clean_cropped.T.ravel() 
    # Target grid for interpolation (from PINN plotting)- interpolate onto the grid defined by x_plot_1d_np and t_plot_1d_np
    
    X_target_mesh, T_target_mesh = np.meshgrid(x_plot_1d_np, t_plot_1d_np)

    print(f"Interpolating FD data onto PINN plotting grid of shape ({nt_plot}, {nx_plot})")

    u_fd_interpolated_flat = griddata(points_fd, values_fd,
                                     (X_target_mesh.ravel(), T_target_mesh.ravel()),
                                     method='cubic', fill_value=0.0)

    # Reshape the interpolated FD data to (nt_plot, nx_plot)
    u_fd_interpolated = u_fd_interpolated_flat.reshape(nt_plot, nx_plot)

    # Plot interpolated FD solution
    plt.figure(figsize=(10,6))
    plt.contourf(x_plot_1d_np, t_plot_1d_np, u_fd_interpolated, levels=50, cmap='viridis')
    plt.colorbar(label='u(x,t) FD solution (interpolated)')
    plt.xlabel('Space x')
    plt.ylabel('Time t')
    plt.title('Finite Difference Solution (Ground Truth, Interpolated)')
    plt.savefig("fd_solution_interpolated.png")
    plt.show()

    # Plot difference (u_pinn_for_plot is (nt,nx), u_fd_interpolated is (nt,nx))
    difference_map = np.abs(u_pinn_for_plot - u_fd_interpolated)
    plt.figure(figsize=(10,6))
    plt.contourf(x_plot_1d_np, t_plot_1d_np, difference_map, levels=50, cmap='Reds')
    plt.colorbar(label='|PINN - FD|')
    plt.xlabel('Space x')
    plt.ylabel('Time t')
    plt.title('Absolute Difference between PINN and FD Solution')
    plt.savefig("pinn_vs_fd_difference.png")
    plt.show()

    # Calculate Relative L2 error
    l2_error = np.linalg.norm(u_pinn_for_plot - u_fd_interpolated) / np.linalg.norm(u_fd_interpolated)
    print(f"Relative L2 error between PINN and FD solution: {l2_error:.4e}")
else:
    print(f"Skipping FD solution comparison: File not found at {fd_data_path}")

# END: Load Finite Difference (FD) solution and Compare
