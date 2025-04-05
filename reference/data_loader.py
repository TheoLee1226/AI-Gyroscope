# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 19:24:15 2025

@author: user
"""


from scipy.io import loadmat
import numpy as np


def load_data():

    data = loadmat(r'gyro_simulation.mat')

    t_data = data['tt'].astype(np.float32)    # MATLAB 模擬時間
    Y_data = data['Y'].astype(np.float32)      # MATLAB 模擬的 Y 數據 (N,7)
    Y0_data = data['Y0'].astype(np.float32)    # MATLAB 模擬初始條件 (1,7)
    phi_data = np.squeeze(data['phi'])         # MATLAB 數據中的 φ
    theta_data = np.squeeze(data['theta'])     # MATLAB 數據中的 θ
    
    # 篩選出 0 ~ 0.1 秒之間的數據
    mask = (t_data >= 0) & (t_data <= 0.1)
    t_data = t_data[mask]
    t_data = t_data.reshape(-1, 1)
    # 修正 mask 維度
    if mask.ndim == 2 and mask.shape[1] == 1:  # 如果是 (8477,1)
        mask = mask.flatten()
    elif mask.ndim == 2 and mask.shape[0] == 1:  # 如果是 (1,8477)
        mask = mask.T.flatten()
    elif mask.ndim == 2 and mask.shape[1] == 7:  # 避免 mask 維度錯誤
        mask = mask[:, 0]  # 取第一列作為 boolean mask
    
    print("mask shape after fix:", mask.shape)
    
    # 確保 mask 是正確的 boolean array
    if mask.shape[0] != Y_data.shape[0]:
        raise ValueError(f"mask shape {mask.shape} does not match Y_data shape {Y_data.shape[0]}")

    
    Y_data = Y_data[mask,:]
    phi_data = phi_data[mask]
    theta_data = theta_data[mask]
    print("t_data shape:", t_data.shape)
    print("Y_data shape:", Y_data.shape)
    print("Y0_data shape:", Y0_data.shape)
    print("phi_data shape:", phi_data.shape)
    print("theta_data shape:", theta_data.shape)
    
    return t_data, Y0_data, Y_data, phi_data, theta_data

if __name__ == "__main__":
    
    t_data, Y0_data, Y_data, phi_data, theta_data = load_data()
    
    
