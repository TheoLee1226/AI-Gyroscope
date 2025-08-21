import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

I_x, I_y, I_z = 1.0, 2.0, 3.0
I = torch.tensor([I_x, I_y, I_z], device=device)

LEARNING_RATE = 1e-3
EPOCHS = 40000 
LAMBDA_PHYSICS = 0.1

def get_ground_truth_torque_torch(t):
    tau_x = 0.5 * torch.sin(t * np.pi)
    tau_y = torch.zeros_like(t)
    tau_z = torch.full_like(t, -0.2)
    return torch.cat([tau_x, tau_y, tau_z], dim=1)

def get_ground_truth_torque_np(t):
    tau_x = 0.5 * np.sin(t * np.pi)
    tau_y = 0.0
    tau_z = -0.2
    return np.array([tau_x, tau_y, tau_z])

def euler_equations_with_torque(t, w, Ix, Iy, Iz):
    wx, wy, wz = w
    tau = get_ground_truth_torque_np(t)
    tau_x, tau_y, tau_z = tau[0], tau[1], tau[2]
    dwx_dt = ((Iy - Iz) * wy * wz + tau_x) / Ix
    dwy_dt = ((Iz - Ix) * wz * wx + tau_y) / Iy
    dwz_dt = ((Ix - Iy) * wx * wy + tau_z) / Iz
    return [dwx_dt, dwy_dt, dwz_dt]

w0 = [1.0, 4.0, 0.5]
t_span = [0, 10]
t_eval = np.linspace(t_span[0], t_span[1], 100)

sol = solve_ivp(
    fun=euler_equations_with_torque, t_span=t_span, y0=w0, t_eval=t_eval,
    args=(I_x, I_y, I_z), dense_output=True, rtol=1e-8, atol=1e-8
)

t_true = torch.tensor(sol.t, dtype=torch.float32).unsqueeze(1).to(device)
omega_true = torch.tensor(sol.y.T, dtype=torch.float32).to(device)
data_points_count = 30
sample_indices = np.random.choice(len(t_true), data_points_count, replace=False)
t_data = t_true[sample_indices]
omega_data = omega_true[sample_indices]

class OmegaNet(nn.Module):
    def __init__(self):
        super(OmegaNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3) # 輸出 omega_x, omega_y, omega_z
        )
    def forward(self, t):
        return self.net(t)

class TorqueNet(nn.Module):
    def __init__(self):
        super(TorqueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3) # 輸出 tau_x, tau_y, tau_z
        )
    def forward(self, t):
        return self.net(t)

omega_net = OmegaNet().to(device)
torque_net = TorqueNet().to(device)

optimizer = torch.optim.Adam(
    list(omega_net.parameters()) + list(torque_net.parameters()), 
    lr=LEARNING_RATE
)
loss_fn = nn.MSELoss()

t_physics = torch.linspace(t_span[0], t_span[1], 200).unsqueeze(1).to(device).requires_grad_(True)

print("開始訓練")
for epoch in range(EPOCHS):
    
    omega_net.train()
    torque_net.train()
    omega_pred_data = omega_net(t_data)
    loss_data = loss_fn(omega_pred_data, omega_data)

    omega_pred_physics = omega_net(t_physics)
    tau_pred_physics = torque_net(t_physics)
    
    dwx_dt = torch.autograd.grad(outputs=omega_pred_physics[:, 0].sum(), inputs=t_physics, create_graph=True)[0]
    dwy_dt = torch.autograd.grad(outputs=omega_pred_physics[:, 1].sum(), inputs=t_physics, create_graph=True)[0]
    dwz_dt = torch.autograd.grad(outputs=omega_pred_physics[:, 2].sum(), inputs=t_physics, create_graph=True)[0]
    
    wx, wy, wz = omega_pred_physics[:, 0], omega_pred_physics[:, 1], omega_pred_physics[:, 2]

    residual_x = (I[0] * dwx_dt.squeeze() + (I[2] - I[1]) * wy * wz) - tau_pred_physics[:, 0]
    residual_y = (I[1] * dwy_dt.squeeze() + (I[0] - I[2]) * wz * wx) - tau_pred_physics[:, 1]
    residual_z = (I[2] * dwz_dt.squeeze() + (I[1] - I[0]) * wx * wy) - tau_pred_physics[:, 2]
    
    loss_physics = loss_fn(residual_x, torch.zeros_like(residual_x)) + \
                   loss_fn(residual_y, torch.zeros_like(residual_y)) + \
                   loss_fn(residual_z, torch.zeros_like(residual_z))
                   
    total_loss = loss_data + LAMBDA_PHYSICS * loss_physics
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Total Loss: {total_loss.item():.6f}, "
              f"Data Loss: {loss_data.item():.6f}, Physics Loss: {loss_physics.item():.6f}")

print("訓練完成!")

omega_net.eval()
torque_net.eval()

with torch.no_grad():
    omega_pred_final = omega_net(t_true).cpu().numpy()
    tau_pred_final = torque_net(t_true).cpu().numpy()

t_true_np = t_true.cpu().numpy()
omega_true_np = omega_true.cpu().numpy()
t_data_np = t_data.cpu().numpy()
omega_data_np = omega_data.cpu().numpy()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), sharex=True)

labels_omega = ['$\omega_x$', '$\omega_y$', '$\omega_z$']
colors = ['r', 'g', 'b']
for i in range(3):
    ax1.plot(t_true_np, omega_true_np[:, i], '-', color=colors[i], label=f'Ground Truth {labels_omega[i]}')
    ax1.plot(t_true_np, omega_pred_final[:, i], '--', color=colors[i], label=f'PINN Prediction {labels_omega[i]}')
ax1.scatter(t_data_np, omega_data_np[:, 0], c='red', s=40, marker='o', edgecolors='k', label='Training Data Points')
ax1.scatter(t_data_np, omega_data_np[:, 1], c='green', s=40, marker='o', edgecolors='k')
ax1.scatter(t_data_np, omega_data_np[:, 2], c='blue', s=40, marker='o', edgecolors='k')
ax1.set_title("PINN Inverse Problem: Angular Velocity Estimation")
ax1.set_ylabel("Angular Velocity (rad/s)")
ax1.legend()
ax1.grid(True)

ground_truth_torque_plot = get_ground_truth_torque_torch(t_true).cpu().numpy()
labels_tau = [r'$\tau_x$', r'$\tau_y$', r'$\tau_z$']
for i in range(3):
    ax2.plot(t_true_np, ground_truth_torque_plot[:, i], '-', color=colors[i], label=f'Ground Truth {labels_tau[i]}')
    ax2.plot(t_true_np, tau_pred_final[:, i], '--', color=colors[i], label=f'PINN Discovered {labels_tau[i]}')
ax2.set_title("PINN Inverse Problem: External Torque Discovery")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Torque (Nm)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("model_fitting_PINNs\\plots\\pinn_gyroscope_torque.png")