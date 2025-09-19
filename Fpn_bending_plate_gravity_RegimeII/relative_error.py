import numpy as np
import csv
import os
from os import path
import pandas as pd
import time

start_time = time.time()
# Function to compute the RBF interpolation coefficient matrix
def podBmtx(Y, X):
    """
    RBF interpolation function using linear spline type.
    Parameters:
    - Y: matrix from POD modal projection, output_matrix
    - X: Parameter matrix
    Returns:
    - B: RBF interpolation coefficient matrix
    """
    # Normalize parameters to [0, 1]
    x = np.zeros_like(X, dtype=float)
    minX = np.min(X, axis=1)
    maxX = np.max(X, axis=1)

    for j in range(X.shape[0]):
        for i in range(X.shape[1]):
            x[j, i] = (X[j, i] - minX[j]) / (maxX[j] - minX[j])

    # Construct G matrix
    n = x.shape[1] #data_size
    G = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            G[i, j] = np.sqrt(np.sum((x[:, i] - x[:, j]) ** 2))

    # Solve for B = Y @ G^-1
    try:
        B = Y @ np.linalg.inv(G)
    except np.linalg.LinAlgError:
        B = Y @ np.linalg.pinv(G)
    return B

# Function to compute the G vector
def podGvec(X, Xtest):
    """
    Construct Gstar vector for given parameters.
    Parameters:
    - X: parameter matrix
    - Xtest: new parameter
    Returns:
    - Gstar:  G vector
    """
    # Normalize parameter matrix P
    a, n = X.shape
    x = np.zeros_like(X, dtype=float)
    normalized_Xtest = np.zeros_like(Xtest, dtype=float)

    minX = np.min(X, axis=1)
    maxX = np.max(X, axis=1)

    for j in range(a):
        for i in range(n):
            x[j, i] = (X[j, i] - minX[j]) / (maxX[j] - minX[j])

    for j in range(Xtest.shape[0]):
        for i in range(Xtest.shape[1]):
            normalized_Xtest[j, i] = (Xtest[j, i] - minX[j]) / (maxX[j] - minX[j])

    Gstar = np.zeros((n, Xtest.shape[1]))

    for i in range(Xtest.shape[1]):
        for k in range(n):
            Gstar[k, i] = np.sqrt(np.sum((normalized_Xtest[:, i] - x[:, k]) ** 2))
    return Gstar

# Function to compute the l 
def determine_components(eigenvalues, threshold=0.99):
    """
    计算保留主成分的最小数量 `l`，使得累积贡献率 >= 阈值
    参数:
        eigenvalues (np.ndarray): 特征值数组（按降序排列）
        threshold (float): 目标累积贡献率阈值（默认 0.99）
    返回:
        l (int): 保留的主成分数量
    """
    total_sum = np.sum(eigenvalues)
    cumulative_sum = np.cumsum(eigenvalues)
    cumulative_ratio = cumulative_sum / total_sum
    
    # 检查是否存在满足条件的索引
    if np.any(cumulative_ratio >= threshold):
        l = np.argmax(cumulative_ratio >= threshold) + 1  # 索引从0开始，需+1
    else:
        l = len(eigenvalues)  # 未满足阈值时保留全部
    
    return l

def format_parameter_name(P, H):
    """将参数转换为文件名友好格式"""
    return f"P_{P:.6f}_H_{H:.6f}".replace('.', 'd')

# 读取参数文件，构造参数矩阵 X 与 Xtest
namespace = {}
with open(r"F:\Fast-pn-actuator\Fpn_bending\GitHub_code\Fpn_bending_plate_gravity_RegimeII\Parameters_P_detaH.py", "r") as file:  # 文件名修改
    exec(file.read(), namespace)
# 构造二维参数矩阵
X = np.array([namespace["training_P_values"], 
            namespace["training_detaH_values"]])# (2, n_train)
Xtest = np.array([namespace["test_P_values"], 
                namespace["test_detaH_values"]])# (2, n_test)

# # 定义数据类型列表（注意：这里"X", "Y", "Z"为坐标数据）
# data_types = ["X", "Y", "Z", "U", "maxPrincipal", "mises", "F"]

# 定义路径
base_path = r'F:\Fast-pn-actuator\Fpn_bending\GitHub_code\Fpn_bending_plate_gravity_RegimeII'
output_folder = os.path.join(base_path, 'output_data')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
training_set_dir = os.path.join(base_path, 'FEModelFiles_training_set')
test_set_dir = os.path.join(base_path, 'FEModelFiles_testing_set')
fa_file_template = os.path.join(output_folder, 'Fa_l.csv')
B_file_template = os.path.join(output_folder, 'B_matrix.csv')



# POD-RBF
Y_train_file = os.path.join(output_folder, 'Y_training.csv')
Y_test_file = os.path.join(output_folder, 'Y_test.csv')
Y = np.loadtxt(Y_train_file, delimiter=',')
Ytest = np.loadtxt(Y_test_file, delimiter=',')

# POD_SVD process
V1, S, V2T = np.linalg.svd(Y)
Fa = V1
m, n = Y.shape
S_matrix = np.zeros((m, n))
S_matrix[:len(S), :len(S)] = np.diag(S)
A = S_matrix @ V2T

# 计算特征值（奇异值的平方）
eigenvalues = S**2
total_eigenvalues = np.sum(eigenvalues)

# 计算不同l值的累积贡献率并输出
cumulative_ratios = []
mse_results = []  # 新增：存储 l 和对应的 MSE

for l in range(1, 16):  # l从1到10
    # 计算累积贡献率
    cumulative_sum = np.sum(eigenvalues[:l])
    ratio = cumulative_sum / total_eigenvalues
    cumulative_ratios.append((l, ratio))
    
    # 截断模态
    Fa_l = Fa[:, :l]
    A_l = A[:l, :]
    
    # RBF 插值预测
    B = podBmtx(A_l, X)      # 计算 B 矩阵
    Gstar = podGvec(X, Xtest) # 计算 G 向量
    Ystar = Fa_l @ B @ Gstar  # 预测 Y*
    
    # 计算均方误差
    mse = np.mean((Ystar - Ytest)**2)
    mse_results.append((l, mse))  # 新增：记录 l 和 MSE

# 将结果保存到CSV文件
contributions_file = os.path.join(output_folder, 'cumulative_contributions.csv')
with open(contributions_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['l', 'Cumulative Contribution Rate'])
    for l, ratio in cumulative_ratios:
        writer.writerow([l, f"{ratio:f}"])

mse_file = os.path.join(output_folder, 'mse_results.csv')
with open(mse_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['l', 'MSE'])
    for l, mse in mse_results:
        writer.writerow([l, f"{mse:f}"])


end_time = time.time()
print(f"\n总运行时间: {end_time - start_time:.2f} 秒")