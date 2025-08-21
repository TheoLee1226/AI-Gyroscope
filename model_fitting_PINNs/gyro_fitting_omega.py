import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

I_x, I_y, I_z = 1.0, 2.0, 3.0
I = torch.tensor([I_x, I_y, I_z], device=device)

LEARNING_RATE = 1e-3
EPOCHS = 20000
LAMBDA_PHYSICS = 0.1 

def euler_equations(t, w, Ix, Iy, Iz):
    wx, wy, wz = w
    dwx_dt = (Iy - Iz) / Ix * wy * wz
    dwy_dt = (Iz - Ix) / Iy * wz * wx
    dwz_dt = (Ix - Iy) / Iz * wx * wy
    return [dwx_dt, dwy_dt, dwz_dt]

w0 = [1.0, 4.0, 0.5]  
t_span = [0, 10]   
t_eval = np.linspace(t_span[0], t_span[1], 100) 

sol = solve_ivp(
    fun=euler_equations,
    t_span=t_span,
    y0=w0,
    t_eval=t_eval,
    args=(I_x, I_y, I_z),
    dense_output=True,
    rtol=1e-8, atol=1e-8
)

t_true = torch.tensor(sol.t, dtype=torch.float32).unsqueeze(1).to(device)
omega_true = torch.tensor(sol.y.T, dtype=torch.float32).to(device)

data_points_count = 30
sample_indices = np.random.choice(len(t_true), data_points_count, replace=False)
t_data = t_true[sample_indices]
omega_data = omega_true[sample_indices]

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3) 
        )

    def forward(self, t):
        return self.net(t)

model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss() 

t_physics = torch.linspace(t_span[0], t_span[1], 200).unsqueeze(1).to(device).requires_grad_(True)

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    
    omega_pred_data = model(t_data)
    loss_data = loss_fn(omega_pred_data, omega_data)

    omega_pred_physics = model(t_physics)
    
    dwx_dt = torch.autograd.grad(outputs=omega_pred_physics[:, 0], inputs=t_physics, grad_outputs=torch.ones_like(omega_pred_physics[:, 0]), create_graph=True)[0]
    dwy_dt = torch.autograd.grad(outputs=omega_pred_physics[:, 1], inputs=t_physics, grad_outputs=torch.ones_like(omega_pred_physics[:, 1]), create_graph=True)[0]
    dwz_dt = torch.autograd.grad(outputs=omega_pred_physics[:, 2], inputs=t_physics, grad_outputs=torch.ones_like(omega_pred_physics[:, 2]), create_graph=True)[0]

    wx, wy, wz = omega_pred_physics[:, 0], omega_pred_physics[:, 1], omega_pred_physics[:, 2]

    residual_x = I[0] * dwx_dt.squeeze() - (I[1] - I[2]) * wy * wz
    residual_y = I[1] * dwy_dt.squeeze() - (I[2] - I[0]) * wz * wx
    residual_z = I[2] * dwz_dt.squeeze() - (I[0] - I[1]) * wx * wy
    
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

model.eval()
with torch.no_grad():
    omega_pred_final = model(t_true).cpu().numpy()

t_true_np = t_true.cpu().numpy()
omega_true_np = omega_true.cpu().numpy()
t_data_np = t_data.cpu().numpy()
omega_data_np = omega_data.cpu().numpy()

print("True Angular Velocities:")
print(omega_true_np[:5])
print("Predicted Angular Velocities:")
print(omega_pred_final[:5])

plt.figure(figsize=(12, 8))
labels = ['$\omega_x$', '$\omega_y$', '$\omega_z$']
colors = ['r', 'g', 'b']

for i in range(3):
    plt.plot(t_true_np, omega_true_np[:, i], '-', color=colors[i], label=f'Ground Truth {labels[i]}')
    plt.plot(t_true_np, omega_pred_final[:, i], '--', color=colors[i], label=f'PINN Prediction {labels[i]}')

plt.scatter(t_data_np, omega_data_np[:, 0], c='red', s=40, marker='o', edgecolors='k', label='Training Data Points')
plt.scatter(t_data_np, omega_data_np[:, 1], c='green', s=40, marker='o', edgecolors='k')
plt.scatter(t_data_np, omega_data_np[:, 2], c='blue', s=40, marker='o', edgecolors='k')

plt.title("PINN for Gyroscope Data Fitting")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.legend()
plt.grid(True)
plt.savefig("model_fitting_PINNs\\plots\\pinn_gyroscope.png")