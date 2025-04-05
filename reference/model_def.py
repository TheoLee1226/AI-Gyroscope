import tensorflow as tf
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.models import load_model
# import numpy as np

class ClipConstraint(Constraint):
    """ 限制變數在 [min_value, max_value] 範圍內 """
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

class Neural_Network(tf.keras.Model):
    """ 定義神經網路模型 """
    def __init__(self, output_dimension=7, hidden_layers=3, neurons_per_layer=32, activation='tanh'):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.output_dimension = output_dimension

        # 定義隱藏層
        self.hidden = [
            tf.keras.layers.Dense(neurons_per_layer, activation=activation, kernel_initializer='glorot_normal')
            for _ in range(hidden_layers)
        ]
        
        # 輸出層
        self.out = tf.keras.layers.Dense(output_dimension)

        # # 定義可訓練變數，並施加 ClipConstraint
        # self.X2 = tf.Variable(50.0, trainable=True, dtype='float32', constraint=ClipConstraint(0.001, 1000.0))
        # self.Im = tf.Variable(50.0, trainable=True, dtype='float32', constraint=ClipConstraint(0.001, 1000.0))
        # self.E0 = tf.Variable(1.0, trainable=True, dtype='float32', constraint=ClipConstraint(0.001, 1000.0))
        # self.E1 = tf.Variable(1.0, trainable=True, dtype='float32', constraint=ClipConstraint(0.001, 1000.0))
        # self.t0 = tf.Variable(1.0, trainable=True, dtype='float32', constraint=ClipConstraint(0.001, 1000))
        # self.t1 = tf.Variable(1.0, trainable=True, dtype='float32', constraint=ClipConstraint(0.001, 1000))

        # self.X2_list, self.Im_list, self.E0_list = [], [], []
        # self.E1_list, self.t0_list, self.t1_list = [], [], []

    def call(self, X):
        """ 前向傳播 """
        Z = X
        for layer in self.hidden:
            Z = layer(Z)
        return self.out(Z)
    
    
#%% 計算微分 (如果需要的話)
def get_derivatives(model, t):
    with tf.GradientTape(persistent=True) as tape:

        tape.watch(t)
        Y_hat = model(t)

    dY_hat_dt = tape.batch_jacobian(Y_hat,t)
    dY_hat_dt = tf.squeeze(dY_hat_dt, axis=-1)  # 變成 (batch_size, 3)
    del tape

    return  Y_hat, dY_hat_dt

#%%
def quat_mul(q1, q2):
    # 四元數相乘，q1, q2: (N,4)
    s1 = q1[:,0:1]
    v1 = q1[:,1:4]
    s2 = q2[:,0:1]
    v2 = q2[:,1:4]
    s = s1*s2 - tf.reduce_sum(v1*v2, axis=1, keepdims=True)
    v = s1*v2 + s2*v1 + tf.linalg.cross(v1, v2)
    return tf.concat([s, v], axis=1)

def quat2mat(q):
    # q: (N,4), 返回旋轉矩陣 R: (N,3,3)
    s = q[:,0:1]
    vx = q[:,1:2]
    vy = q[:,2:3]
    vz = q[:,3:4]
    R11 = 1 - 2*vy**2 - 2*vz**2
    R12 = 2*vx*vy - 2*s*vz
    R13 = 2*vx*vz + 2*s*vy
    R21 = 2*vx*vy + 2*s*vz
    R22 = 1 - 2*vx**2 - 2*vz**2
    R23 = 2*vy*vz - 2*s*vx
    R31 = 2*vx*vz - 2*s*vy
    R32 = 2*vy*vz + 2*s*vx
    R33 = 1 - 2*vx**2 - 2*vy**2
    R1 = tf.concat([R11, R12, R13], axis=1)
    R2 = tf.concat([R21, R22, R23], axis=1)
    R3 = tf.concat([R31, R32, R33], axis=1)
    R = tf.stack([R1, R2, R3], axis=1)
    return R  # shape: (N,3,3)

#%% 計算損失
def compute_all_losses(model,t, Y_real, Y0_real, var_list):
    
    Y_hat, dY_hat_dt = get_derivatives(model, t)
    
    
    q = Y_hat[:, 0:4] 
    w = Y_hat[:, 4:7]
    dq_dt = dY_hat_dt[:, 0:4]   # 计算 q 相对于 t 的梯度
    dw_dt = dY_hat_dt[:, 4:7]   # 计算 w 相对于 t 的梯度
    
    M = var_list[0]          # 質量
    R0 = var_list[1]         # 陀螺半徑（參考值）
    a = var_list[2]          # 長度參數（重心距離支點）
    g = var_list[3]          # 重力加速度
    I0 = var_list[4] 
    invI0 = var_list[5]      # I0反矩陣
    
    # 計算所有loss
    # 計算旋轉矩陣 R 及其
    R = quat2mat(q)  # R旋轉矩陣
    RT = tf.transpose(R, perm=[0, 2, 1])  # R旋轉矩陣轉置
    
    # 定義力矩 tau = cross( R*[0,0,a], M*g*[0,0,-1] )
    v_a = tf.constant([[0.0, 0.0, a]], dtype=tf.float32)  # 轉軸(1,3)
    force = tf.constant([[0.0, 0.0, -M*g]], dtype=tf.float32)  # 重力(1,3)
    v_a_batch = tf.tile(v_a, [tf.shape(R)[0], 1]) # 每個時間點的轉軸

    R_v = tf.matmul(R, tf.expand_dims(v_a_batch, -1)) # 每個時間點的轉軸受旋轉矩陣變化而變化
    R_v = tf.squeeze(R_v, axis=-1) #轉完矩陣後留下變(N,3,1)要轉乘(N.3)
    tau = tf.linalg.cross(R_v, tf.tile(force, [tf.shape(R)[0], 1])) #算出力矩tau = R_v x F
    
    # 計算 R * I0 * R^T
    I0_tf = tf.constant(I0, dtype=tf.float32) # 轉動慣量
    invI0_tf = tf.constant(invI0, dtype=tf.float32) # 反轉動慣量
    R_I0_RT = tf.matmul(tf.matmul(R, I0_tf), RT) #座標轉換的轉動慣量
    w_exp = tf.expand_dims(w, -1) #多增一軸以利於後續與矩陣運算
    L = tf.matmul(R_I0_RT, w_exp) # 角動量 L=R_I0_RT*w (3,1)
    L = tf.squeeze(L, axis=-1) # 角動量 L=R_I0_RT*w (3,)
    
    
    # 預測 dq/dt 的物理部分： dq/dt = 0.5 * quat_mul( [0, w], q )
    zero_w = tf.zeros([tf.shape(w)[0], 1], dtype=tf.float32)
    w_quat = tf.concat([zero_w, w], axis=1) # 把w用成四元數(0,wx,wy,wz)
    
    # 計算殘差：網路預測的導數與物理預測的差異
    res_q = dq_dt - 0.5 * quat_mul(w_quat, q) # dq/dt=(1/2)*q.w_euat
    res_w = dw_dt - tf.linalg.matvec(invI0_tf, (tau - tf.linalg.cross(w, L))) # R_I0_RT*dw/dt + w x L= tau
    Y = tf.concat([q, w], axis=1)
    Y_tmin = Y[0, :]
    
    r1_loss = Y_real - Y_hat
    r2_loss = Y0_real - Y_tmin
    r3_loss = res_q
    r4_loss = res_w
    
    rq_loss = Y_real[:,0:4] - Y_hat[:,0:4]
    rw_loss = Y_real[:,4:7] - Y_hat[:,4:7]
    
    loss1  =  1*tf.reduce_sum(tf.square(r1_loss)) 
    loss2  =  1*tf.reduce_mean(tf.square(r2_loss)) 
    loss3  =  0.01*tf.reduce_mean(tf.square(r3_loss)) 
    loss4  =  0.01*tf.reduce_mean(tf.square(r4_loss)) 
    
    lossq  =  1*tf.reduce_mean(tf.square(rq_loss)) 
    lossw  =  1*tf.reduce_mean(tf.square(rw_loss)) 
    
    return loss1, loss2, loss3, loss4, lossq, lossw, Y_hat, dY_hat_dt


#%% 計算梯度
def get_gradients(model,t, Y_real, Y0_real, var_list):
    
    with tf.GradientTape(persistent=True) as tape:
        # This tape is for derivatives with respect to trainable variables
        tape.watch(model.trainable_variables)
        loss1, loss2, loss3, loss4, lossq, lossw, Y_hat, dY_hat_dt = compute_all_losses(model,t, Y_real, Y0_real, var_list)
        loss_T = loss1 + loss2 + loss3 + loss4
    g1 = tape.gradient(loss_T, model.trainable_variables)

    del tape

    return loss1, loss2, loss3, loss4, lossq, lossw, loss_T, Y_hat, dY_hat_dt, g1

#%% 創建模型
def create_model(input_dimension=1, output_dimension=7, hidden_layers=3, neurons_per_layer=32, activation='tanh'):
    """ 創建並回傳模型 """
    model = Neural_Network(output_dimension, hidden_layers, neurons_per_layer, activation)
    model.build(input_shape=(None, input_dimension))
    return model

#%% 載入模型
def restore_model(path):
    # 載入已訓練的模型
    model = load_model(path)
    print("Model restored successfully.")

    # # 重新限制變數的範圍 (如果需要的話)
    # model.X2 = tf.Variable(model.X2.numpy(), trainable=True, dtype='float32',constraint=ClipConstraint(0.001, 1000))
    # model.Im = tf.Variable(model.Im.numpy(), trainable=True, dtype='float32',constraint=ClipConstraint(0.001, 1000))
    # model.E0 = tf.Variable(model.E0.numpy(), trainable=True, dtype='float32',constraint=ClipConstraint(0.001, 1000))
    # model.E1 = tf.Variable(model.E1.numpy(), trainable=True, dtype='float32',constraint=ClipConstraint(0.001, 1000))
    # model.t0 = tf.Variable(model.t0.numpy(), trainable=True, dtype='float32',constraint=ClipConstraint(0.001, 1000))
    # model.t1 = tf.Variable(model.t1.numpy(), trainable=True, dtype='float32',constraint=ClipConstraint(0.001, 1000))
    
    return model

#%%
if __name__ == "__main__":
    model = create_model()
    print("模型已建立")
