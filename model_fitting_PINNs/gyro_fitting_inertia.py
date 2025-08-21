import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


I_x, I_y, I_z = 1.0, 1.0, 2.0
I_np = np.array([I_x, I_y, I_z]) 
I_true_torch = torch.tensor([I_x, I_y, I_z], device=device, dtype=torch.float32) 

MASS = 0.5
GRAVITY_ACCEL = -9.81
r_cm_body = np.array([0.0, 0.1, 0.1])
g_lab = np.array([0.0, 0.0, GRAVITY_ACCEL])


LEARNING_RATE = 1e-3
EPOCHS = 20000
LAMBDA_PHYSICS = 0.1


def quaternion_to_rotation_matrix(q):
    q0, q1, q2, q3 = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])

def full_dynamics(t, y, Ix, Iy, Iz, m, g_vec_lab, r_cm):
    w = y[:3] # Angular velocity \omega
    q = y[3:] # Quaternion q
    
    R = quaternion_to_rotation_matrix(q)
    g_body = R @ g_vec_lab
    tau_g = np.cross(r_cm, m * g_body)
    
    dwx_dt = ((Iy - Iz) * w[1] * w[2] + tau_g[0]) / Ix
    dwy_dt = ((Iz - Ix) * w[2] * w[0] + tau_g[1]) / Iy
    dwz_dt = ((Ix - Iy) * w[0] * w[1] + tau_g[2]) / Iz
    dw_dt = np.array([dwx_dt, dwy_dt, dwz_dt])

    q0, q1, q2, q3 = q
    wx, wy, wz = w
    dq0_dt = 0.5 * (-q1*wx - q2*wy - q3*wz)
    dq1_dt = 0.5 * ( q0*wx + q2*wz - q3*wy)
    dq2_dt = 0.5 * ( q0*wy - q1*wz + q3*wx)
    dq3_dt = 0.5 * ( q0*wz + q1*wy - q2*wx)
    dq_dt = np.array([dq0_dt, dq1_dt, dq2_dt, dq3_dt])
    
    return np.concatenate((dw_dt, dq_dt))

w0 = np.array([5.0, 1.0, 1.0])
q0 = np.array([1.0, 0.0, 0.0, 0.0])
y0 = np.concatenate((w0, q0))

t_span = [0, 10]
t_eval = np.linspace(t_span[0], t_span[1], 200)
sol = solve_ivp(
    fun=full_dynamics, t_span=t_span, y0=y0, t_eval=t_eval,
    args=(I_x, I_y, I_z, MASS, g_lab, r_cm_body),
    dense_output=True, rtol=1e-6, atol=1e-6
)

t_true_np = sol.t
omega_true_np = sol.y[:3, :].T
q_true_np = sol.y[3:, :].T


tau_g_true_np = np.zeros_like(omega_true_np)
for i, (t, q) in enumerate(zip(t_true_np, q_true_np)):
    R = quaternion_to_rotation_matrix(q)
    g_body = R @ g_lab
    tau_g_true_np[i] = np.cross(r_cm_body, MASS * g_body)

interp_tau_x = interp1d(t_true_np, tau_g_true_np[:, 0], kind='cubic', fill_value="extrapolate")
interp_tau_y = interp1d(t_true_np, tau_g_true_np[:, 1], kind='cubic', fill_value="extrapolate")
interp_tau_z = interp1d(t_true_np, tau_g_true_np[:, 2], kind='cubic', fill_value="extrapolate")

def get_known_torque_torch(t):
    t_np = t.detach().cpu().numpy()
    tau_x = interp_tau_x(t_np)
    tau_y = interp_tau_y(t_np)
    tau_z = interp_tau_z(t_np)
    tau = np.stack([tau_x, tau_y, tau_z], axis=1).squeeze()
    return torch.tensor(tau, dtype=torch.float32).to(device)


t_true = torch.tensor(t_true_np, dtype=torch.float32).unsqueeze(1).to(device)
omega_true = torch.tensor(omega_true_np, dtype=torch.float32).to(device)

data_points_count = 40
sample_indices = np.random.choice(len(t_true), data_points_count, replace=False)
t_data = t_true[sample_indices]
omega_data = omega_true[sample_indices]

class OmegaNet(nn.Module):
    def __init__(self):
        super(OmegaNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 3) # 輸出為 [\omega_x, \omega_y, \omega_z]
        )
    def forward(self, t):
        return self.net(t)

model = OmegaNet().to(device)
loss_fn = nn.MSELoss()

I_learned = nn.Parameter(torch.tensor([1.5, 1.5, 1.5], device=device, dtype=torch.float32))

optimizer = torch.optim.Adam(
    list(model.parameters()) + [I_learned],
    lr=LEARNING_RATE
)

t_physics = torch.linspace(t_span[0], t_span[1], 300).unsqueeze(1).to(device).requires_grad_(True)

print("開始訓練以推斷轉動慣量 (Starting training to infer moments of inertia)...")
for epoch in range(EPOCHS):
    model.train()
    
    omega_pred_data = model(t_data)
    loss_data = loss_fn(omega_pred_data, omega_data)

    omega_pred_physics = model(t_physics)
    
    dw_dt_list = []
    for i in range(3):

        dw_dt_i = torch.autograd.grad(
            outputs=omega_pred_physics[:, i].sum(), 
            inputs=t_physics, 
            create_graph=True
        )[0]
        dw_dt_list.append(dw_dt_i)
    
    dwx_dt, dwy_dt, dwz_dt = dw_dt_list[0], dw_dt_list[1], dw_dt_list[2]
    wx, wy, wz = omega_pred_physics[:, 0], omega_pred_physics[:, 1], omega_pred_physics[:, 2]
    
    known_tau = get_known_torque_torch(t_physics)
    
    residual_x = (I_learned[0] * dwx_dt.squeeze() + (I_learned[2] - I_learned[1]) * wy * wz) - known_tau[:, 0]
    residual_y = (I_learned[1] * dwy_dt.squeeze() + (I_learned[0] - I_learned[2]) * wz * wx) - known_tau[:, 1]
    residual_z = (I_learned[2] * dwz_dt.squeeze() + (I_learned[1] - I_learned[0]) * wx * wy) - known_tau[:, 2]
    
    loss_physics = loss_fn(residual_x, torch.zeros_like(residual_x)) + \
                   loss_fn(residual_y, torch.zeros_like(residual_y)) + \
                   loss_fn(residual_z, torch.zeros_like(residual_z))

    total_loss = loss_data + LAMBDA_PHYSICS * loss_physics
    
    # 反向傳播與優化
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # 定期打印訓練進度
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Total Loss: {total_loss.item():.6f}, "
              f"Data Loss: {loss_data.item():.6f}, Physics Loss: {loss_physics.item():.6f}")
        print(f"  -> Learned I: [{I_learned[0].item():.4f}, {I_learned[1].item():.4f}, {I_learned[2].item():.4f}]")


# --- 訓練完成後，顯示結果 ---
print("\n訓練完成 (Training Complete)!")
print("-" * 50)
print(f"真實轉動慣量 (Ground Truth I): [{I_x:.4f}, {I_y:.4f}, {I_z:.4f}]")
# 使用 .data 取得張量數據，避免梯度追蹤
print(f"PINN 推斷的轉動慣量 (Inferred I): [{I_learned.data[0].item():.4f}, {I_learned.data[1].item():.4f}, {I_learned.data[2].item():.4f}]")
print("-" * 50)

# --- 結果可視化 ---
model.eval()
with torch.no_grad():
    omega_pred_final_np = model(t_true).cpu().numpy()

output_dir = os.path.join(os.path.dirname(__file__), "plots")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), sharex=True)

labels_omega = ['$\omega_x$', '$\omega_y$', '$\omega_z$']
colors = ['r', 'g', 'b']
for i in range(3):
    ax1.plot(t_true_np, omega_true_np[:, i], '-', color=colors[i], label=f'Ground Truth {labels_omega[i]}')
    ax1.plot(t_true_np, omega_pred_final_np[:, i], '--', color=colors[i], label=f'PINN Prediction {labels_omega[i]}')

ax1.scatter(t_data.cpu().numpy(), omega_data.cpu().numpy()[:, 0], c='red', s=40, marker='o', edgecolors='k', label='Training Data Points')
ax1.scatter(t_data.cpu().numpy(), omega_data.cpu().numpy()[:, 1], c='green', s=40, marker='o', edgecolors='k')
ax1.scatter(t_data.cpu().numpy(), omega_data.cpu().numpy()[:, 2], c='blue', s=40, marker='o', edgecolors='k')

ax1.set_title("PINN Fitting of Gyroscope and Inertia Inference")
ax1.set_ylabel("Angular Velocity (rad/s)")
ax1.legend()
ax1.grid(True)

labels_tau = [r'$\tau_x$', r'$\tau_y$', r'$\tau_z$']
for i in range(3):
    ax2.plot(t_true_np, tau_g_true_np[:, i], '-', color=colors[i], label=f'Known Gravity Torque {labels_tau[i]}')

ax2.set_title("Input Gravity Torque (Known to PINN via Interpolation)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Torque (Nm)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()

plt.savefig(os.path.join(output_dir, "pinn_inertia_inference.png"))
print(f"\n結果圖已儲存至 '{output_dir}/pinn_inertia_inference.png'")

