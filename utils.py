import numpy as np

def normalizer(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # norm_data = numerator / (denominator + 1e-7)
    norm_data = numerator / denominator
    return norm_data, np.min(data, 0), np.max(data, 0)




# 这个 slicing_window 函数的作用是对输入的二维数据进行滑动窗口切片，生成多个连续的子数组。它常用于时间序列分析、机器学习等领域，尤其是需要将序列数据转换为固定长度的子序列作为输入的场景（例如在 LSTM 或其他时序模型中）。
def slicing_window(data, n_in):
    list_of_features = []

    for i in range(len(data)-n_in+1):
        arr_features = data[i:(i+n_in), :]
        list_of_features.append(arr_features)

    return np.array(list_of_features)

def batch_generator_with_time(data, time, batch_size):
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]         
    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)
    return X_mb, T_mb



# random_generator 函数的作用是生成一个批次的随机噪声向量，每个噪声向量的维度是 z_dim，并且根据给定的时间步长 T_mb 填充噪声序列，
# 确保每个样本的噪声长度符合要求。如果某个样本的时间步长小于 max_seq_len，则其余的时间步会用零向量填充。这些噪声向量通常用作生成对抗网络（GAN）生成器的输入，
# 用来生成假数据。
def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    Z_mb = list()
    for i in range(batch_size):
        temp = np.zeros([max_seq_len, z_dim])
        temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
        temp[:T_mb[i],:] = temp_Z
        Z_mb.append(temp_Z)
    return Z_mb

