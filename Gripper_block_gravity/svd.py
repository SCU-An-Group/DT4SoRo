import numpy as np
import csv
import os
import time
import pandas as pd

start_time = time.time()

# =========================================================
# 工具函数
# =========================================================

def determine_components(eigenvalues, threshold=0.999):
    """根据累计贡献率确定主成分数量 l"""
    total_sum = np.sum(eigenvalues)
    cumulative_sum = np.cumsum(eigenvalues)
    cumulative_ratio = cumulative_sum / total_sum
    if np.any(cumulative_ratio >= threshold):
        l = np.argmax(cumulative_ratio >= threshold) + 1
    else:
        l = len(eigenvalues)
    return l

def format_parameter_name(P, H):
    """将参数转换为文件名友好格式"""
    return f"P_{P:.6f}_H_{H:.6f}"

def cdist_cols(A, B=None):
    """按列计算欧氏距离矩阵"""
    if B is None:
        B = A
    A2 = np.sum(A * A, axis=0, keepdims=True)  # 1×n
    B2 = np.sum(B * B, axis=0, keepdims=True)  # 1×m
    cross = A.T @ B                            # n×m
    D2 = A2.T + B2 - 2.0 * cross
    D2 = np.maximum(D2, 0.0)
    return np.sqrt(D2)

def normalize_with_ranges(X, Xtest, ranges, dims, on_out_of_range='warn+clip'):
    """用指定全局范围归一化；零跨度维度会被忽略"""
    min_arr = np.array([ranges[d][0] for d in dims], dtype=float).reshape(-1, 1)
    max_arr = np.array([ranges[d][1] for d in dims], dtype=float).reshape(-1, 1)
    span = max_arr - min_arr
    valid_mask = (span[:, 0] > 0)
    span_safe = np.where(span == 0.0, 1.0, span)

    Xn = (X - min_arr) / span_safe
    Xtestn = (Xtest - min_arr) / span_safe

    if on_out_of_range in ('clip', 'warn+clip'):
        Xn = np.clip(Xn, 0.0, 1.0)
        Xtestn = np.clip(Xtestn, 0.0, 1.0)

    if on_out_of_range in ('warn', 'warn+clip'):
        def _warn(name, Z):
            if np.any(Z < 0.0) or np.any(Z > 1.0):
                print(f"[WARN] {name} 存在超出全局范围的值（已{('裁剪' if 'clip' in on_out_of_range else '未裁剪')}）。")
        _warn('X_norm', Xn)
        _warn('Xtest_norm', Xtestn)

    if not np.any(valid_mask):
        raise ValueError("所有参数维度的全局跨度均为0，至少需要一个有变化的参数。")

    return Xn, Xtestn, valid_mask

# =========================================================
# 主程序
# =========================================================

# 读取参数文件
namespace = {}
with open(r"F:\PYC\Gripper\Parameters_P_detaH.py", "r") as file:
    exec(file.read(), namespace)

# 构造二维参数矩阵
X = np.array([namespace["training_P_values"],
              namespace["training_detaH_values"]])  # (2, n_train)
Xtest = np.array([namespace["test_P_values"],
                  namespace["test_detaH_values"]])  # (2, n_test)

# ------------- 策略 A: 指定全局范围 -----------------
# ⚠️ 这里需要你自己根据工程情况设置
PARAM_RANGES = {
    'P': (min(X[0, :].min(), Xtest[0, :].min()), 
          max(X[0, :].max(), Xtest[0, :].max())),
    'H': (10.0, 50.0),   # 举例：虽然数据全是20.0，但指定全局范围
}
dims = ['P', 'H']

# 归一化
X_norm, Xtest_norm, valid_mask = normalize_with_ranges(X, Xtest, PARAM_RANGES, dims)
X_use = X_norm[valid_mask, :]
Xtest_use = Xtest_norm[valid_mask, :]

# 预计算 G, Gstar
G = cdist_cols(X_use)
Gstar = cdist_cols(X_use, Xtest_use)
try:
    G_factor = np.linalg.inv(G)
except np.linalg.LinAlgError:
    G_factor = np.linalg.pinv(G)

# =========================================================
# 数据准备
# =========================================================

data_types = ["X", "Y", "Z", "U", "maxPrincipal", "mises", "F"]
base_path = r'F:\PYC\Gripper'
output_folder = os.path.join(base_path, 'output_data')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

test_P_values = namespace["test_P_values"]
num_test = Xtest.shape[1]

fa_file_template = os.path.join(output_folder, 'Fa_l_{}.csv')
B_file_template = os.path.join(output_folder, 'B_matrix_{}.csv')

predictions = {}
mse_summary = {}

# =========================================================
# 循环每个物理量
# =========================================================

for dtype in data_types:
    # 加载数据
    Y_train_file = os.path.join(output_folder, f'Y_training_{dtype}.csv')
    Y_test_file = os.path.join(output_folder, f'Y_test_{dtype}.csv')
    Y = np.loadtxt(Y_train_file, delimiter=',')
    Ytest = np.loadtxt(Y_test_file, delimiter=',')

    # SVD
    V1, S, V2T = np.linalg.svd(Y)
    Fa = V1
    m, n = Y.shape
    S_matrix = np.zeros((m, n))
    S_matrix[:len(S), :len(S)] = np.diag(S)
    A = np.dot(S_matrix, V2T)

    # 自动选择 l
    # eigenvalues = S**2
    # l = determine_components(eigenvalues, threshold=0.9999)
    l = 6
    print(f"For dtype {dtype}: l = {l}")

    # 截断
    Fa_l = Fa[:, :l]
    A_l = A[:l, :]

    # RBF 插值预测
    B = A_l @ G_factor
    Ystar = Fa_l @ (B @ Gstar)

    # 保存 Fa_l, B
    np.savetxt(fa_file_template.format(dtype), Fa_l, delimiter=',', fmt='%f')
    np.savetxt(B_file_template.format(dtype), B, delimiter=',', fmt='%f')

    # MSE
    mse = np.mean((Ystar - Ytest)**2)
    print(f"Mean Squared Error for {dtype}: {mse}")
    mse_summary[dtype] = mse

    predictions[dtype] = Ystar

# =========================================================
# 合并结果 & 保存
# =========================================================

header = ['Node Label'] + data_types
num_nodes = list(predictions.values())[0].shape[0]

for j in range(Xtest.shape[1]):
    P_val = Xtest[0, j]
    H_val = Xtest[1, j]
    param_str = format_parameter_name(P_val, H_val)
    combined = np.zeros((num_nodes, 1 + len(data_types)))
    combined[:, 0] = np.arange(1, num_nodes + 1)
    for i, dtype in enumerate(data_types):
        combined[:, i+1] = predictions[dtype][:, j]
    output_path = os.path.join(output_folder, f'Y_prediction_all_{param_str}.csv')
    np.savetxt(output_path, combined, delimiter=',', fmt='%f',
               header=",".join(header), comments='')
    print(f"Combined prediction for {param_str} saved to {output_path}")

# 保存误差汇总
pd.DataFrame([mse_summary]).to_csv(os.path.join(output_folder, "MSE_summary.csv"), index=False)

end_time = time.time()
print(f"\n总运行时间: {end_time - start_time:.2f} 秒")
