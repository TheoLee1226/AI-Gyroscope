#%% 載入需要函數庫
import os
import tensorflow as tf
import numpy as np
from time import time
import matplotlib.pyplot as plt
from data_loader import load_data
from model_def import get_gradients, create_model,  restore_model

#%% 定義並轉成tensorflow 靜態圖
@tf.function
def train_step():
    # 這行如果一直出現代表一直在retrace graph (會導致速度變慢)
    print('[Side Effect] Retracing graph')
    
    # Compute current loss and gradient w.r.t. parameters
    loss1, loss2, loss3, loss4, lossq, lossw, loss_T, Y_hat, dY_hat_dt, g1 = \
        get_gradients(model,t_data,Y_data,Y0_data, \
                      [M,R0,a,g,I0,invI0])
    
    # Perform gradient descent step
    optim.apply_gradients(zip(g1, model.trainable_variables))

    return loss1, loss2, loss3, loss4, lossq, lossw, loss_T, Y_hat, dY_hat_dt

#%% 訓練過程
def train(N = 20000):
    loss_history = []
    # Start timer
    start_time = time()
    for j in range(N+1):
        
        loss1, loss2, loss3, loss4, lossq, lossw, loss_T, Y_hat, dY_hat_dt = train_step()
    
        # Output current loss after 5000 iterates
        if j%5000 == 0:
            elapsed = time() - start_time
            # Append current loss to loss_history
            loss_history.append(loss_T.numpy())
    
            print(f'Elapsed Time: {elapsed:.2f}')    
            print('It: {:05d}| Total loss = {:6.2e}| Data loss = {:6.2e}| Quaternion Loss = {:6.2e}| Angular Velocity Loss = {:6.2e}|'.format(j,loss_T, loss1, lossq, lossw, ))
            print('IC Loss = {:6.2e}|  PDE Loss [1] = {:6.2e}|  PDE Loss [2] = {:6.2e}|'.format(loss2, loss3, loss4))
            start_time = time()
    return  loss_history, Y_hat, dY_hat_dt

T0 = time()

#%% 載入數據
t_data, Y0_data, Y_data, phi_data, theta_data = load_data()

t_data = tf.constant(t_data, dtype='float32')
Y0_data = tf.constant(Y0_data, dtype='float32')
Y_data = tf.constant(Y_data, dtype='float32')

M = 1.0          # 質量
R0 = 1.0         # 陀螺半徑（參考值）
a = 2.0          # 長度參數（重心距離支點）
g = 9.8          # 重力加速度

# 陀螺的慣性張量 I0 (3x3)， I0 = 2/5*M*R0^2*eye(3) + M*a^2*[1 0 0; 0 1 0; 0 0 0]
I0 = np.array([[2/5*M*R0**2 + M*a**2, 0                   , 0           ],
               [0                   , 2/5*M*R0**2 + M*a**2, 0           ],
               [0                   ,  0                  , 2/5*M*R0**2]])
invI0 = np.linalg.inv(I0) # I0反矩陣

#%% 創建模型
model = create_model(input_dimension=1, output_dimension=7, hidden_layers=3, neurons_per_layer=16, activation='tanh')
optim = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)


#%% 載入已訓練好的模型 (如果需要的話)
# model = restore_model('SHG_model')
# # 強制將所有層設置為可訓練 (避免儲存模型的時候是不能訓練的，這樣載入後會都不能訓練)
# for layer in model.layers:
#     layer.trainable = True  
# # model.summary() 

# # 強制將某幾層設置為不可訓練
# for layer in model.layers[:-2]:
#     layer.trainable = False  
# model.compile()
# for layer in model.layers:
#     print(layer.name, layer.trainable)
    

#%% 訓練
epoch = 100000  # Number of training epochs
loss_history, Y_hat, dY_hat_dt = train(epoch)


#%% 儲存訓練好的模型
# model.save('Precession_model')


#%% Prediction with the new (test) data
def quat2mat_np(q):
    s, vx, vy, vz = q[0], q[1], q[2], q[3]
    R = np.array([[1 - 2*vy**2 - 2*vz**2,    2*vx*vy - 2*s*vz,      2*vx*vz + 2*s*vy],
                  [2*vx*vy + 2*s*vz,          1 - 2*vx**2 - 2*vz**2,  2*vy*vz - 2*s*vx],
                  [2*vx*vz - 2*s*vy,          2*vy*vz + 2*s*vx,      1 - 2*vx**2 - 2*vy**2]])
    return R
# 可根據需求進一步分析角速度 w_pred 或與數值模擬做比較
def quaternion_to_euler(q):
    """
    根據 MATLAB 方法計算 進動角 φ 和 傾斜角 θ
    q: (N,4) 四元數數組，格式為 [s, x, y, z]
    """
    phi_list = []
    theta_list = []

    for i in range(q.shape[0]):  # 遍歷每個時間點
        R = quat2mat_np(q[i])  # 轉換為旋轉矩陣
        phi = np.arctan2(R[1,2], R[0,2])  # 進動角 φ
        theta = np.arccos(np.clip(R[2,2], -1.0, 1.0))  # 傾斜角 θ
        phi_list.append(phi)
        theta_list.append(theta)

    return np.array(phi_list), np.array(theta_list)


# t_star = t_data
# Y_hat = model(t_star) 
q_pred = Y_hat[:, 0:4].numpy()
w_pred = Y_hat[:, 4:7].numpy()

q_data = Y_data[:, 0:4].numpy()
w_data = Y_data[:, 4:7].numpy()

#%% 提取 q_pred 和 q_data 
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
q_labels = ['q0', 'q1', 'q2', 'q3']
colors = ['r', 'g', 'b', 'm']

for i in range(4):
    row = i // 2
    col = i % 2
    axs[row, col].plot(t_data, q_data[:, i], 'r-', label=f'Data {q_labels[i]}')
    axs[row, col].plot(t_data, q_pred[:, i], 'b--', label=f'Predicted {q_labels[i]}')
    axs[row, col].set_ylabel(q_labels[i])
    axs[row, col].legend()
    axs[row, col].grid(True)

axs[1, 0].set_xlabel('Time (s)')
axs[1, 1].set_xlabel('Time (s)')
fig.suptitle('Quaternion Components Comparison (q0 ~ q3)', fontsize=16)
plt.tight_layout()
plt.show()

# 角速度
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
w_labels = ['w0', 'w1', 'w2']

for i in range(3):
    axs[i].plot(t_data, w_data[:, i], 'r-', label=f'Data {w_labels[i]}')
    axs[i].plot(t_data, w_pred[:, i], 'b--', label=f'Predicted {w_labels[i]}')
    axs[i].set_ylabel(w_labels[i])
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel('Time (s)')
fig.suptitle('Angular Velocity Components Comparison')
plt.tight_layout()
plt.show()

#%% 提取 q 和 w

# 計算 dq/dt 和 dw/dt（數值微分）
# 使用 np.diff 計算相鄰元素的差異，這樣得到的就是數值導數
dt = np.diff(t_data, axis=0)  # (N-1,1)
dq0_dt_data = np.diff(q_data, axis=0)  / dt # 沿著時間軸微分
dw0_dt_data = np.diff(w_data, axis=0)  / dt # 沿著時間軸微分

# 由於 np.diff 會減少一個時間點，這裡我們可以手動添加一個初始點（假設為0）
dq0_dt_data = np.concatenate([np.zeros((1, 4), dtype=np.float32), dq0_dt_data], axis=0)
dw0_dt_data = np.concatenate([np.zeros((1, 3), dtype=np.float32), dw0_dt_data], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(t_data, dq0_dt_data[:,0], 'r-', label='Data dq_dt')
plt.plot(t_data, dY_hat_dt[:,0], 'b--', label='Predicted dq_dt')
plt.xlabel('Time (s)')
plt.ylabel('w0')
plt.legend()
plt.title('Comparison of dq_dt vs Time')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t_data, dw0_dt_data[:,0], 'r-', label='Data dw_dt')
plt.plot(t_data, dY_hat_dt[:,4:5], 'b--', label='Predicted dw_dt')
plt.xlabel('Time (s)')
plt.ylabel('w0')
plt.legend()
plt.title('Comparison of dw_dt vs Time')
plt.tight_layout()
plt.show()









