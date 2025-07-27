import numpy as np
import gyro_real_word_simualtion as rws

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class simulation_dataset(Dataset):

    def __init__(self, train=True):
        if train:
            try:
                self.data_loading = np.load('model_fitting\\dataset\\training_gyro_simulation_data.npz')
                print("training_gyro_simulation_data.npz loaded")
                self.mean = np.mean(self.data_loading['data'], axis=(0, 1))
                self.std = np.std(self.data_loading['data'], axis=(0, 1))
                data_normalized = (self.data_loading['data'] - self.mean) / self.std
                print("Data normalized")
                self.std[self.std == 0] = 1e-6
                np.savez('model_fitting\\dataset\\gyro_simulation_data_normalization_parameter.npz', std =self.std, mean=self.mean)
            except FileNotFoundError:
                print("training_gyro_simulation_data.npz not found")
                return 
        else:
            try:
                self.data_loading = np.load('model_fitting\\dataset\\validation_gyro_simulation_data.npz')
                print("validation_gyro_simulation_data.npz loaded")
                try:
                    normalization_parameter = np.load('model_fitting\\dataset\\gyro_simulation_data_normalization_parameter.npz')
                    self.mean = normalization_parameter['mean']
                    self.std = normalization_parameter['std']
                    print("Normalization parameters loaded")
                    data_normalized = (self.data_loading['data'] - self.mean) / self.std
                    print("Data normalized")
                except FileNotFoundError:
                    print("Normalization parameters not found")
                    return
            except FileNotFoundError:
                print("validation_gyro_simulation_data.npz not found")
                return

        print("Data shape: ", self.data_loading['data'].shape)
        self.data_simulation = torch.tensor(data_normalized, dtype=torch.float32)
        print("Data simulation shape: ", self.data_simulation.shape)
        self.I = torch.tensor(self.data_loading['I'], dtype=torch.float32)
        print("I shape: ", self.I.shape)
        self.M = torch.tensor(self.data_loading['M'], dtype=torch.float32)
        print("M shape: ", self.M.shape)
        self.g = torch.tensor(self.data_loading['g'], dtype=torch.float32)
        print("g shape: ", self.g.shape)
        self.H = torch.tensor(self.data_loading['H'], dtype=torch.float32)
        print("H shape: ", self.H.shape)
        self.X_0 = torch.tensor(self.data_loading['X_0'], dtype=torch.float32)
        print("X_0 shape: ", self.X_0.shape)
        self.D_X_0 = torch.tensor(self.data_loading['D_X_0'], dtype=torch.float32)
        print("D_X_0 shape: ", self.D_X_0.shape)
        print("Data loaded successfully")

    def __len__(self):
        return len(self.data_loading['data'])

    def __getitem__(self, idx):
        data_simulation = self.data_simulation[idx]
        I = self.I[idx]
        M = self.M[idx]
        g = self.g[idx]
        H = self.H[idx]
        X_0 = self.X_0[idx]
        D_X_0 = self.D_X_0[idx]
        return data_simulation, I, M, g, H, X_0, D_X_0  

    def create_dataset(self ,total_time ,sample_num ,data_num, I_limit, M_limit, g_limit, H_limit, X_0_limit, D_X_0_limit):
        I_1 = np.random.uniform(I_limit[0][0], I_limit[0][1], size=(data_num, 1))
        I_2 = np.random.uniform(I_limit[1][0], I_limit[1][1], size=(data_num, 1))
        I_3 = np.random.uniform(I_limit[2][0], I_limit[2][1], size=(data_num, 1))
        I = np.hstack((I_1, I_2, I_3))

        M = np.random.uniform(M_limit[0], M_limit[1], size=(data_num, 1))
        g = np.random.uniform(g_limit[0], g_limit[1], size=(data_num, 1))
        H = np.random.uniform(H_limit[0], H_limit[1], size=(data_num, 1))

        Theta_0 = np.random.uniform(X_0_limit[0][0], X_0_limit[0][1], size=(data_num, 1))
        Phi_0 = np.random.uniform(X_0_limit[1][0], X_0_limit[1][1], size=(data_num, 1))
        Psi_0 = np.random.uniform(X_0_limit[2][0], X_0_limit[2][1], size=(data_num, 1))
        X_0 = np.hstack((Theta_0, Phi_0, Psi_0))

        Theta_dot_0 = np.random.uniform(D_X_0_limit[0][0], D_X_0_limit[0][1], size=(data_num, 1))
        Phi_dot_0 = np.random.uniform(D_X_0_limit[1][0], D_X_0_limit[1][1], size=(data_num, 1))
        Psi_dot_0 = np.random.uniform(D_X_0_limit[2][0], D_X_0_limit[2][1], size=(data_num, 1))
        D_X_0 = np.hstack((Theta_dot_0, Phi_dot_0, Psi_dot_0))

        time = np.linspace(0, total_time, sample_num)

        simulation = rws.real_word_simulation()

        output = []
        for i in tqdm(range(data_num), desc="Simulating data"):
            output.append(simulation.real_word_simulation(time, I[i], X_0[i], D_X_0[i], M[i], g[i], H[i]))

        return np.array(output), I, M, g, H, X_0, D_X_0

    def create_training_dataset(self, total_time, sample_num, data_num, I_limit, M_limit, g_limit, H_limit, X_0_limit, D_X_0_limit):
        print("Creating training data...")
        data, I, M, g, H, X_0, D_X_0 = self.create_dataset(total_time, sample_num, data_num, I_limit, M_limit, g_limit, H_limit, X_0_limit, D_X_0_limit)
        np.savez('model_fitting\\dataset\\training_gyro_simulation_data.npz', data=data, I=I, M=M, g=g, H=H, X_0=X_0, D_X_0=D_X_0)
        print("Data saved to training_gyro_simulation_data.npz")

    def create_validation_dataset(self, total_time, sample_num, data_num, I_limit, M_limit, g_limit, H_limit, X_0_limit, D_X_0_limit):
        print("Creating validation data...")
        data, I, M, g, H, X_0, D_X_0 = self.create_dataset(total_time, sample_num, data_num, I_limit, M_limit, g_limit, H_limit, X_0_limit, D_X_0_limit)
        np.savez('model_fitting\\dataset\\validation_gyro_simulation_data.npz', data=data, I=I, M=M, g=g, H=H, X_0=X_0, D_X_0=D_X_0)
        print("Data saved to validation_gyro_simulation_data.npz")

if __name__ == "__main__":
    dataset = simulation_dataset(train=True)
    
    dataset.create_training_dataset(total_time=10, sample_num=1000, data_num=100000,
                                 I_limit=[(0.1, 0.5), (0.1, 0.5), (0.1, 0.5)], 
                                 M_limit=(0.1, 0.5), g_limit=(9.81, 9.81), H_limit=(0.1, 0.5), 
                                 X_0_limit=[(-np.pi/2, np.pi/2), (0, 0), (0, 0)], 
                                 D_X_0_limit=[(-1, 1), (-1, 1), (-1, 1)])
    dataset.create_validation_dataset(total_time=10, sample_num=1000, data_num=1000,
                                   I_limit=[(0.1, 0.5), (0.1, 0.5), (0.1, 0.5)], 
                                   M_limit=(0.1, 0.5), g_limit=(9.81, 9.81), H_limit=(0.1, 0.5), 
                                   X_0_limit=[(-np.pi/2, np.pi/2), (0, 0), (0, 0)], 
                                   D_X_0_limit=[(-1, 1), (-1, 1), (-1, 1)])
    
    
    