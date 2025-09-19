import numpy as np
import csv
import os
from os import path


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


newPara = [0.07]


# 定义输出文件夹（保存预测结果以及pod_mode）
base_path = r'F:/Fast-pn-actuator/Fpn_bending/noBoard/Gravity'
output_folder = os.path.join(base_path, 'output_data')

# 读取参数文件，构造参数矩阵 X 与 Xpred
namespace = {}
with open(r"F:\Fast-pn-actuator\Fpn_bending\noBoard\Gravity\Parameters_P.py", "r") as file:
    exec(file.read(), namespace)
X = np.array(namespace["P_values"]).reshape(1, -1)
Xpred = np.array(newPara).reshape(1, -1)
num_test = Xpred.shape[1]  # 测试参数数量

# 用于存放预测结果的字典
predictions = {}
# 定义数据类型列表（注意：这里"X", "Y", "Z"为坐标数据）
data_types = ["X", "Y", "Z", "U", "mises"]

for dtype in data_types:
    # 读取Fa_l以及B矩阵
    Fa_l_file = os.path.join(output_folder, 'Fa_l_{}.csv'.format(dtype))
    B_file = os.path.join(output_folder, 'B_matrix_{}.csv'.format(dtype))
    
    # 加载数据（使用 np.loadtxt 读取 CSV 文件）
    Fa_l = np.loadtxt(Fa_l_file, delimiter=',')
    B = np.loadtxt(B_file, delimiter=',')
    
    # RBF 插值预测
    Gstar = podGvec(X, Xpred)
    Ypred = np.dot(Fa_l, np.dot(B, Gstar))
    
    # 保存预测结果矩阵到字典中（仍需保留）
    predictions[dtype] = Ypred

# 将同一测试参数 P 下的各数据类型预测结果合并到同一文件中
# 定义输出文件头，假设“Node Label”为节点编号，从1开始
header = ['Node Label'] + data_types
# 假设所有预测结果矩阵的行数一致（即节点数相同）
num_nodes = list(predictions.values())[0].shape[0]

for j in range(num_test):
    # 构造一个空数组用于存放合并结果，第一列为节点编号，其余列依次为各数据类型预测结果
    combined = np.zeros((num_nodes, 1 + len(data_types)))
    # 填入节点编号（假设从1开始）
    combined[:, 0] = np.arange(1, num_nodes + 1)
    # 对于每个数据类型，填入对应的预测列
    for i, dtype in enumerate(data_types):
        # predictions[dtype]的第 j 列
        combined[:, i+1] = predictions[dtype][:, j]
    
    # 构造输出文件名，例如：Y_prediction_all_P_<P值>.csv
    file_name = 'Y_anyPara_pred_all_P_{}.csv'.format(newPara[j])
    output_path = os.path.join(output_folder, file_name)
    # 使用 np.savetxt 写入 CSV 文件，同时写入 header（设置 comments='' 以避免行首出现#）
    np.savetxt(output_path, combined, delimiter=',', fmt='%f',
               header=",".join(header), comments='')
    print("Combined prediction for P = {} saved to {}".format(newPara[j], output_path))