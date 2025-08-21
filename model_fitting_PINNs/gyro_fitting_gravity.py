import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

# --- 0. 環境設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置 (Using device): {device}")

# --- 1. 物理與訓練參數 ---
# 轉動慣量
I_x, I_y, I_z = 1.0, 2.0, 3.0
I_np = np.array([I_x, I_y, I_z])
I_torch = torch.tensor([I_x, I_y, I_z], device=device)

# --- 新增的物理參數 ---
MASS = 0.5  # kg
GRAVITY_ACCEL = -9.81 # m/s^2, 沿 Z 軸負方向
# 質心在物體座標系中的位置向量 (質心不在支點上才會產生力矩)
r_cm_body = np.array([0.0, 0.1, 0.1]) 
# 實驗室座標系中的重力向量
g_lab = np.array([0.0, 0.0, GRAVITY_ACCEL])

# 訓練超參數
LEARNING_RATE = 1e-3
EPOCHS = 30000
LAMBDA_PHYSICS = 0.1

# --- 2. 數據生成：完整的剛體動力學模型 ---
# 輔助函數：從四元數計算旋轉矩陣 (NumPy 版本)
def quaternion_to_rotation_matrix(q):
    q0, q1, q2, q3 = q / np.linalg.norm(q) # 正規化
    return np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])

# 完整的動力學 ODE 函數 (給 solve_ivp 使用)
def full_dynamics(t, y, Ix, Iy, Iz, m, g_vec_lab, r_cm):
    # y 是一個 7 維的狀態向量: [wx, wy, wz, q0, q1, q2, q3]
    w = y[:3]
    q = y[3:]
    
    # 1. 計算姿態 -> 旋轉矩陣 -> 力矩
    R = quaternion_to_rotation_matrix(q)
    g_body = R @ g_vec_lab # 將重力向量從實驗室座標轉換到物體座標
    tau_g = np.cross(r_cm, m * g_body) # 計算重力力矩
    
    # 2. 計算角速度的變化 (歐拉方程式)
    dwx_dt = ((Iy - Iz) * w[1] * w[2] + tau_g[0]) / Ix
    dwy_dt = ((Iz - Ix) * w[2] * w[0] + tau_g[1]) / Iy
    dwz_dt = ((Ix - Iy) * w[0] * w[1] + tau_g[2]) / Iz
    dw_dt = np.array([dwx_dt, dwy_dt, dwz_dt])
    
    # 3. 計算四元數的變化 (運動學方程)
    q0, q1, q2, q3 = q
    wx, wy, wz = w
    dq0_dt = 0.5 * (-q1*wx - q2*wy - q3*wz)
    dq1_dt = 0.5 * ( q0*wx + q2*wz - q3*wy)
    dq2_dt = 0.5 * ( q0*wy - q1*wz + q3*wx)
    dq3_dt = 0.5 * ( q0*wz + q1*wy - q2*wx)
    dq_dt = np.array([dq0_dt, dq1_dt, dq2_dt, dq3_dt])
    
    return np.concatenate((dw_dt, dq_dt))

# 初始條件
w0 = np.array([5.0, 1.0, 1.0])      # 初始角速度
q0 = np.array([1.0, 0.0, 0.0, 0.0]) # 初始姿態 (無旋轉)
y0 = np.concatenate((w0, q0))

# 解 ODE
t_span = [0, 10]
t_eval = np.linspace(t_span[0], t_span[1], 200) # 生成更密集的點用於插值
sol = solve_ivp(
    fun=full_dynamics, t_span=t_span, y0=y0, t_eval=t_eval,
    args=(I_x, I_y, I_z, MASS, g_lab, r_cm_body),
    dense_output=True, rtol=1e-6, atol=1e-6
)

# 從解中提取數據
t_true_np = sol.t
omega_true_np = sol.y[:3, :].T
q_true_np = sol.y[3:, :].T

# 計算每個時間點的真實重力力矩
tau_g_true_np = np.zeros_like(omega_true_np)
for i, (t, q) in enumerate(zip(t_true_np, q_true_np)):
    R = quaternion_to_rotation_matrix(q)
    g_body = R @ g_lab
    tau_g_true_np[i] = np.cross(r_cm_body, MASS * g_body)

# --- 3. 建立力矩的插值函數 (我們的 "Oracle") ---
print("建立力矩的插值函數...")
# 使用 interp1d 建立一個可以給出任意時間點力矩的函數
interp_tau_x = interp1d(t_true_np, tau_g_true_np[:, 0], kind='cubic', fill_value="extrapolate")
interp_tau_y = interp1d(t_true_np, tau_g_true_np[:, 1], kind='cubic', fill_value="extrapolate")
interp_tau_z = interp1d(t_true_np, tau_g_true_np[:, 2], kind='cubic', fill_value="extrapolate")

# 建立一個 PyTorch wrapper 函數，以便在訓練中使用
def get_known_torque_torch(t):
    t_np = t.detach().cpu().numpy()
    tau_x = interp_tau_x(t_np)
    tau_y = interp_tau_y(t_np)
    tau_z = interp_tau_z(t_np)
    tau = np.stack([tau_x, tau_y, tau_z], axis=1).squeeze()
    return torch.tensor(tau, dtype=torch.float32).to(device)

# --- 4. 準備 PINN 的訓練數據 ---
t_true = torch.tensor(t_true_np, dtype=torch.float32).unsqueeze(1).to(device)
omega_true = torch.tensor(omega_true_np, dtype=torch.float32).to(device)
data_points_count = 40
sample_indices = np.random.choice(len(t_true), data_points_count, replace=False)
t_data = t_true[sample_indices]
omega_data = omega_true[sample_indices]

# --- 5. PINN 模型定義 (沿用之前的 OmegaNet) ---
class OmegaNet(nn.Module):
    def __init__(self):
        super(OmegaNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 3)
        )
    def forward(self, t):
        return self.net(t)

# --- 6. 訓練流程 (與「已知外力矩」版本幾乎一樣) ---
model = OmegaNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

t_physics = torch.linspace(t_span[0], t_span[1], 300).unsqueeze(1).to(device).requires_grad_(True)

print("開始訓練 (使用由重力產生的複雜力矩)...")
for epoch in range(EPOCHS):
    model.train()
    
    # 數據損失
    omega_pred_data = model(t_data)
    loss_data = loss_fn(omega_pred_data, omega_data)

    # 物理損失
    omega_pred_physics = model(t_physics)
    dwx_dt = torch.autograd.grad(outputs=omega_pred_physics[:, 0].sum(), inputs=t_physics, create_graph=True)[0]
    dwy_dt = torch.autograd.grad(outputs=omega_pred_physics[:, 1].sum(), inputs=t_physics, create_graph=True)[0]
    dwz_dt = torch.autograd.grad(outputs=omega_pred_physics[:, 2].sum(), inputs=t_physics, create_graph=True)[0]
    
    wx, wy, wz = omega_pred_physics[:, 0], omega_pred_physics[:, 1], omega_pred_physics[:, 2]

    # 從插值函數獲取已知的重力力矩
    known_tau = get_known_torque_torch(t_physics)
    
    # 計算物理殘差 (LHS - RHS)
    residual_x = (I_torch[0] * dwx_dt.squeeze() + (I_torch[2] - I_torch[1]) * wy * wz) - known_tau[:, 0]
    residual_y = (I_torch[1] * dwy_dt.squeeze() + (I_torch[0] - I_torch[2]) * wz * wx) - known_tau[:, 1]
    residual_z = (I_torch[2] * dwz_dt.squeeze() + (I_torch[1] - I_torch[0]) * wx * wy) - known_tau[:, 2]
    
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

# --- 7. 結果可視化 ---
model.eval()
with torch.no_grad():
    omega_pred_final_np = model(t_true).cpu().numpy()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), sharex=True)

# 子圖 1: 角速度擬合結果
labels_omega = ['$\omega_x$', '$\omega_y$', '$\omega_z$']
colors = ['r', 'g', 'b']
for i in range(3):
    ax1.plot(t_true_np, omega_true_np[:, i], '-', color=colors[i], label=f'Ground Truth {labels_omega[i]}')
    ax1.plot(t_true_np, omega_pred_final_np[:, i], '--', color=colors[i], label=f'PINN Prediction {labels_omega[i]}')
ax1.scatter(t_data.cpu().numpy(), omega_data.cpu().numpy()[:, 0], c='red', s=40, marker='o', edgecolors='k', label='Training Data Points')
ax1.scatter(t_data.cpu().numpy(), omega_data.cpu().numpy()[:, 1], c='green', s=40, marker='o', edgecolors='k')
ax1.scatter(t_data.cpu().numpy(), omega_data.cpu().numpy()[:, 2], c='blue', s=40, marker='o', edgecolors='k')
ax1.set_title("PINN Fitting of Gyroscope under Gravity")
ax1.set_ylabel("Angular Velocity (rad/s)")
ax1.legend()
ax1.grid(True)

# 子圖 2: 已知的重力力矩 (作為物理約束的輸入)
labels_tau = [r'$\tau_x$', r'$\tau_y$', r'$\tau_z$']
for i in range(3):
    ax2.plot(t_true_np, tau_g_true_np[:, i], '-', color=colors[i], label=f'Known Gravity Torque {labels_tau[i]}')
ax2.set_title("Input Gravity Torque (Known to PINN via Interpolation)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Torque (Nm)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()

# 儲存圖片
output_dir = "model_fitting_PINNs/plots"
file_name = "pinn_gyroscope_gravity_torque.png"
full_path = os.path.join(output_dir, file_name)
os.makedirs(output_dir, exist_ok=True)
plt.savefig(full_path)
print(f"圖片已儲存至: {full_path}")

plt.show()