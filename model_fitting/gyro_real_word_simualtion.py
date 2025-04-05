import numpy as np
from scipy.integrate import solve_ivp

class real_word_simulation:
    def __init__(self):
        pass

    @staticmethod
    def real_word_simulation_solve_ivp_fun(t, y, I, M, g, H):
        I_12 = I[0]
        I_3 = I[2]

        theta = y[0]
        phi = y[1]
        psi = y[2]

        theta_dot = y[3]
        phi_dot = y[4]
        psi_dot = y[5]

        M_val = M[0]
        g_val = g[0]
        H_val = H[0]

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        epsilon = 1e-10
        if abs(sin_theta) < epsilon:
            safe_sin_theta = np.sign(sin_theta) * epsilon + (1 - np.sign(sin_theta) ** 2) * epsilon
            if abs(safe_sin_theta) < epsilon:
                safe_sin_theta = epsilon
        else:
            safe_sin_theta = sin_theta

        omega_30 = phi_dot * cos_theta + psi_dot

        theta_dot_dot = phi_dot**2 * sin_theta * cos_theta - (I_3 / I_12) * phi_dot * sin_theta * omega_30 + (M_val * g_val * H_val / I_12) * sin_theta
        phi_dot_dot = (theta_dot / (I_12 * safe_sin_theta)) * (I_3 * omega_30 - 2 * I_12 * phi_dot * cos_theta)
        psi_dot_dot = phi_dot * theta_dot * sin_theta - phi_dot_dot * cos_theta

        return [theta_dot, phi_dot, psi_dot, theta_dot_dot, phi_dot_dot, psi_dot_dot]

    def real_word_simulation(self, time, I, X_0, D_X_0, M, g, H):

        y = [X_0[0], X_0[1], X_0[2], D_X_0[0], D_X_0[1], D_X_0[2]]
        y_0 = np.array([X_0[0], X_0[1], X_0[2], D_X_0[0], D_X_0[1], D_X_0[2]])

        output = solve_ivp(self.real_word_simulation_solve_ivp_fun, [time[0], time[-1]], y_0, t_eval=time, args=(I, M, g, H), method='LSODA', rtol=1e-6, atol=1e-6)
        if output.success == False:
            print("Error: ", output.message)
        
        return np.array(output.y).T
