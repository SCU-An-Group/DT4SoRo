import numpy as np
import csv
import os
from os import path
import time

start_time = time.time()

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
        B = np.dot(Y, np.linalg.inv(G))
    except np.linalg.LinAlgError:
        B = np.dot(Y, np.linalg.pinv(G))
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
def determine_components(eigenvalues, threshold=0.999):
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
    return f"P_{P:.6f}_H_{H:.6f}"


# 读取参数文件，构造参数矩阵 X 与 Xtest
namespace = {}
with open(r"C:/Pu_file/Board_new_new/Parameters_P_detaH_new_new.py", "r") as file:  # 文件名修改
    exec(file.read(), namespace)
# 构造二维参数矩阵
X = np.array([namespace["training_P_values"], 
            namespace["training_detaH_values"]])# (2, n_train)
Xtest = np.array([namespace["test_P_values"], 
                namespace["test_detaH_values"]])# (2, n_test)

# 定义数据类型列表（注意：这里"X", "Y", "Z"为坐标数据）
data_types = ["X", "Y", "Z", "U", "mises"]

# 定义输出文件夹（保存预测结果以及pod_mode）
base_path = r'C:/Pu_file/Board_new_new'
output_folder = os.path.join(base_path, 'output_data')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 测试参数列表（假设 test_P_values 中的每一项对应一列）
test_P_values = namespace["test_P_values"]
num_test = Xtest.shape[1]  # 测试参数数量

# 新增：定义输出Fa_l和B矩阵的文件名前缀
fa_file_template = os.path.join(output_folder, 'Fa_l_{}.csv')
B_file_template = os.path.join(output_folder, 'B_matrix_{}.csv')

# 用于存放预测结果的字典
predictions = {}

for dtype in data_types:
    # 构造训练集与测试集文件名
    Y_train_file = os.path.join(output_folder, 'Y_training_{}.csv'.format(dtype))
    Y_test_file = os.path.join(output_folder, 'Y_test_{}.csv'.format(dtype))
    
    # 加载数据（使用 np.loadtxt 读取 CSV 文件）
    Y = np.loadtxt(Y_train_file, delimiter=',')
    Ytest = np.loadtxt(Y_test_file, delimiter=',')
    
    # POD-SVD 分解
    V1, S, V2T = np.linalg.svd(Y)
    Fa = V1
    m, n = Y.shape
    S_matrix = np.zeros((m, n))
    S_matrix[:len(S), :len(S)] = np.diag(S)
    A = np.dot(S_matrix, V2T)
    
    # 确定截断的主成分数量 l
    # eigenvalues = S**2  # 计算特征值（奇异值的平方）
    # l = determine_components(eigenvalues, threshold=0.9999)
    l = 6
    print("For dtype {}: l = {}".format(dtype, l))
    
    # 截断 SVD 结果
    Fa_l = Fa[:, :l]
    A_l = A[:l, :]
    
    # RBF 插值预测
    B = podBmtx(A_l, X)
    Gstar = podGvec(X, Xtest)
    Ystar = np.dot(Fa_l, np.dot(B, Gstar))
    
    # 立即保存Fa_l矩阵到文件
    current_fa_file = fa_file_template.format(dtype)
    np.savetxt(current_fa_file, Fa_l, delimiter=',', fmt='%f')
    print("Fa_l matrix for {} saved to {}".format(dtype, current_fa_file))
    # 立即保存B矩阵到文件
    current_B_file = B_file_template.format(dtype)
    np.savetxt(current_B_file, B, delimiter=',', fmt='%f')
    print("B matrix for {} saved to {}".format(dtype, current_B_file))

    
    # 删除临时变量以释放内存
    del Fa_l, B  # 删除后这些变量不再占用内存
    
    # 可选：计算均方误差
    mse = np.mean((Ystar - Ytest)**2)
    print("Mean Squared Error for {}: {}".format(dtype, mse))
    
    # 保存预测结果矩阵到字典中（仍需保留）
    predictions[dtype] = Ystar

# 将同一测试参数 P 下的各数据类型预测结果合并到同一文件中
# 定义输出文件头，假设“Node Label”为节点编号，从1开始
header = ['Node Label'] + data_types
# 假设所有预测结果矩阵的行数一致（即节点数相同）
num_nodes = list(predictions.values())[0].shape[0]

for j in range(Xtest.shape[1]):
    P_val = Xtest[0,j]
    H_val = Xtest[1,j]
    param_str = format_parameter_name(P_val, H_val)
    # 构造一个空数组用于存放合并结果，第一列为节点编号，其余列依次为各数据类型预测结果
    combined = np.zeros((num_nodes, 1 + len(data_types)))
    # 填入节点编号（假设从1开始）
    combined[:, 0] = np.arange(1, num_nodes + 1)
    # 对于每个数据类型，填入对应的预测列
    for i, dtype in enumerate(data_types):
        # predictions[dtype]的第 j 列
        combined[:, i+1] = predictions[dtype][:, j]
    
    # 构造输出文件名，例如：Y_prediction_all_P_<P值>.csv
    file_name = 'Y_prediction_all_{}.csv'.format(param_str)
    output_path = os.path.join(output_folder, file_name)
    # 使用 np.savetxt 写入 CSV 文件，同时写入 header（设置 comments='' 以避免行首出现#）
    np.savetxt(output_path, combined, delimiter=',', fmt='%f',
               header=",".join(header), comments='')
    print("Combined prediction for {} saved to {}".format(param_str, output_path))

end_time = time.time()
print(f"\n总运行时间: {end_time - start_time:.2f} 秒")