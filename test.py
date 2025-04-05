import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# --- 物理參數 (Physical Parameters) ---
m = 0.5      # 質量 (kg) - Mass
g = 9.81     # 重力加速度 (m/s^2) - Gravity
r = 0.1      # 支點到質心的距離 (m) - Distance from pivot to Center of Mass
I1 = 0.01  # 繞橫向軸的轉動慣量 (kg*m^2) - Moment of inertia about transverse axis
I3 = 0.005    # 繞對稱軸的轉動慣量 (kg*m^2) - Moment of inertia about symmetry axis

# --- 初始條件 (Initial Conditions) ---
theta0 = np.radians(60)  # 初始傾斜角 (radians) - Initial tilt angle
phi0 = 0.0               # 初始進動角 (radians) - Initial precession angle
psi0 = 0.0               # 初始自旋角 (radians) - Initial spin angle

# 初始角速度 (Initial Angular Velocities)
# 設定初始自旋速度 ω_s (圍繞自身對稱軸)
omega_s_initial = 50.0  # 初始自旋角速度 (rad/s) - Initial spin angular velocity
# 假設初始時只有自旋，且進動和章動的角速度為 0
# ψ_dot ≈ ω_s (當 θ_dot 和 φ_dot 很小時)
# Note: 更精確地說，ω_s 是 ψ_dot + φ_dot*cos(θ)，但常直接設定 ψ_dot(0)
psi_dot0 = omega_s_initial
phi_dot0 = 0.0           # 初始進動角速度 (rad/s) - Initial precession angular velocity
theta_dot0 = 0.0         # 初始章動角速度 (rad/s) - Initial nutation angular velocity

# --- 計算守恆量 (Calculate Conserved Quantities) ---
# L3: 沿對稱軸的角動量分量 (Angular momentum along symmetry axis)
L3 = I3 * (phi_dot0 * np.cos(theta0) + psi_dot0)
# Lz: 沿垂直 Z 軸的角動量分量 (Angular momentum along vertical Z axis)
Lz = I1 * phi_dot0 * np.sin(theta0)**2 + L3 * np.cos(theta0)

# --- 運動方程式 (Equations of Motion) ---
# 定義微分方程組 d(state)/dt = f(t, state)
# state = [theta, theta_dot, phi, psi]
def gyroscope_derivs(t, state):
    theta, theta_dot, phi, psi = state

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # 防止 sin(theta) 接近 0 時的數值問題
    if abs(sin_theta) < 1e-8:
        # 如果 theta 接近 0 或 pi，理論上行為會不同，這裡做個近似處理或報錯
        # 在 theta=0 附近，進動公式失效。真實陀螺儀若垂直則不進動（不穩定）。
        # 為避免除零錯誤，可以返回 0 或設置一個極限
        if abs(theta) < 1e-8: # 接近垂直向上
             return [theta_dot, 0, 0, L3 / I3] # 假設無進動，僅自旋
        else: # 接近垂直向下 (theta 接近 pi)
             return [theta_dot, 0, 0, L3 / I3] # 假設無進動，僅自旋

    # --- 計算角速度 (Calculate Angular Velocities) ---
    # φ_dot (進動角速度)
    phi_dot = (Lz - L3 * cos_theta) / (I1 * sin_theta**2)
    # ψ_dot (自旋相關角速度)
    psi_dot = (L3 / I3) - phi_dot * cos_theta

    # --- 計算二階導數 (Calculate Second Derivatives) ---
    # θ_ddot (章動角加速度)
    term1 = phi_dot**2 * cos_theta * sin_theta # I1*phi_dot^2*sin*cos / I1
    term2 = (L3 / I1) * phi_dot * sin_theta     # I3*ω3*phi_dot*sin / I1 = L3*phi_dot*sin / I1
    term3 = (m * g * r / I1) * sin_theta
    theta_ddot = term1 - term2 + term3

    # 返回狀態變數的導數
    # d(theta)/dt = theta_dot
    # d(theta_dot)/dt = theta_ddot
    # d(phi)/dt = phi_dot
    # d(psi)/dt = psi_dot
    return [theta_dot, theta_ddot, phi_dot, psi_dot]

# --- 數值積分 (Numerical Integration) ---
t_span = (0, 10)  # 模擬時間範圍 (秒) - Simulation time span
dt = 0.01         # 輸出時間間隔 - Output time step
t_eval = np.arange(t_span[0], t_span[1], dt) # 要求解的時間點

initial_state = [theta0, theta_dot0, phi0, psi0] # 初始狀態向量

# 使用 solve_ivp 求解
# 'method='RK45' 是預設值，通常效果很好
# 'dense_output=True' 可以獲得連續的解（插值函數）
sol = solve_ivp(gyroscope_derivs, t_span, initial_state, t_eval=t_eval, dense_output=True, rtol=1e-6, atol=1e-8)

# 從解中提取結果
times = sol.t
theta, theta_dot, phi, psi = sol.y

# --- 計算陀螺儀頂端軌跡 (Calculate Tip Trajectory) ---
# 假設陀螺儀軸長度為 L (用於視覺化)
axis_length = r * 1.5
x_tip = axis_length * np.sin(theta) * np.cos(phi)
y_tip = axis_length * np.sin(theta) * np.sin(phi)
z_tip = axis_length * np.cos(theta)



# --- 靜態繪圖 (Static Plotting) ---
plt.rcParams['font.family'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 繪製軌跡
ax.plot(x_tip, y_tip, z_tip, label='陀螺儀頂端軌跡 (Tip Trajectory)')

# 繪製最後位置的陀螺儀軸線
final_idx = -1
axis_x = [0, x_tip[final_idx]]
axis_y = [0, y_tip[final_idx]]
axis_z = [0, z_tip[final_idx]]
ax.plot(axis_x, axis_y, axis_z, 'r-', linewidth=2, label=f'最終時刻陀螺儀軸 (Axis at t={times[final_idx]:.1f}s)')
ax.scatter([0], [0], [0], c='k', marker='o', s=50, label='支點 (Pivot)') # 標示支點

# 設定座標軸
max_range = axis_length * 1.1
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([0, max_range * 1.1]) # Z 軸從 0 開始較符合直覺
ax.set_xlabel('X 軸')
ax.set_ylabel('Y 軸')
ax.set_zlabel('Z 軸 (垂直向上)')
ax.set_title('陀螺儀進動與章動模擬軌跡\nGyroscope Precession and Nutation Simulation')
ax.legend()
ax.grid(True)
ax.view_init(elev=20., azim=-45) # 調整視角

plt.show()

# --- (可選) 動畫製作 (Optional Animation) ---
fig_anim = plt.figure(figsize=(10, 8))
ax_anim = fig_anim.add_subplot(111, projection='3d')

max_range_anim = axis_length * 1.1
ax_anim.set_xlim([-max_range_anim, max_range_anim])
ax_anim.set_ylim([-max_range_anim, max_range_anim])
ax_anim.set_zlim([0, max_range_anim * 1.1])
ax_anim.set_xlabel('X 軸')
ax_anim.set_ylabel('Y 軸')
ax_anim.set_zlabel('Z 軸 (垂直向上)')
ax_anim.set_title('陀螺儀運動動畫\nGyroscope Motion Animation')
ax_anim.grid(True)
ax_anim.view_init(elev=20., azim=-45)

# 繪製軌跡 (背景)
ax_anim.plot(x_tip, y_tip, z_tip, ':', lw=1, color='gray', label='軌跡')
# 陀螺儀軸線的線物件
line, = ax_anim.plot([], [], [], 'r-', linewidth=3, label='陀螺儀軸')
# 支點
pivot = ax_anim.scatter([0], [0], [0], c='k', marker='o', s=50, label='支點')
# 時間文字
time_text = ax_anim.text2D(0.05, 0.95, '', transform=ax_anim.transAxes)

# 動畫更新函數
def update_anim(frame):
    idx = frame
    # 更新軸線位置
    axis_x_anim = [0, x_tip[idx]]
    axis_y_anim = [0, y_tip[idx]]
    axis_z_anim = [0, z_tip[idx]]
    line.set_data(axis_x_anim, axis_y_anim)
    line.set_3d_properties(axis_z_anim)
    # 更新時間顯示
    time_text.set_text(f'時間 (Time): {times[idx]:.2f}s')
    return line, time_text

# 計算動畫幀數和間隔
skip_frames = 5 # 每隔幾幀繪製一次，加快動畫速度
num_frames = len(times) // skip_frames
interval = dt * skip_frames * 1000 # ms

# 建立動畫物件
ani = animation.FuncAnimation(fig_anim, update_anim, frames=num_frames,
                              interval=interval, blit=True, repeat=False)

# 顯示動畫 (如果需要保存成 gif 或 mp4，需要額外安裝 ffmpeg 或 imagemagick)
# try:
#     ani.save('gyroscope_animation.gif', writer='imagemagick', fps=15)
#     print("動畫已保存為 gyroscope_animation.gif")
# except Exception as e:
#     print(f"保存動畫失敗: {e}. 請確保已安裝 imagemagick 並配置好 matplotlib.")



plt.legend()
plt.show()