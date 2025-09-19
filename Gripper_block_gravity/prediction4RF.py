# -*- coding: utf-8 -*-
"""
Predict RF at a single new parameter using pre-trained RBF coefficients
"""

import numpy as np
import os

# ========== 工具函数 ==========
def normalize_with_ranges(X, Xtest, ranges, dims, on_out_of_range='warn+clip'):
    """
    用指定全局范围归一化；零跨度维度会被忽略
    X: (a, n_train)
    Xtest: (a, n_test)
    ranges: dict, 每个维度的 (min, max)
    dims: list[str], 参数名顺序
    """
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


def podGvec(X, Xtest, ranges, dims):
    """构造 G* 向量，使用安全归一化"""
    Xn, Xtestn, valid_mask = normalize_with_ranges(X, Xtest, ranges, dims)
    Xn_use = Xn[valid_mask, :]
    Xtestn_use = Xtestn[valid_mask, :]

    n = Xn_use.shape[1]
    Gstar = np.zeros((n, Xtestn_use.shape[1]))
    for i in range(Xtestn_use.shape[1]):
        for k in range(n):
            Gstar[k, i] = np.linalg.norm(Xtestn_use[:, i] - Xn_use[:, k])
    return Gstar


# ========== 主程序 ==========
base_path = r'F:\Fast-pn-actuator\Fpn_bending\Gripper'
param_file = os.path.join(base_path, "Parameters_P_detaH.py")
output_folder = os.path.join(base_path, 'output_data')

# 读取参数
namespace = {}
with open(param_file) as f:
    exec(f.read(), namespace)

X = np.array([namespace["training_P_values"],
              namespace["training_detaH_values"]])  # (2, n_train)

# 读取 B 矩阵
B = np.loadtxt(os.path.join(output_folder, "B_matrix_RF.csv"), delimiter=",")

# 定义新的参数点
newPara = np.array([[0.079], [20.0]])  # shape (2,1)
P_val, detaH_val = newPara.ravel()

# 全局归一化区间（可手动设定）
PARAM_RANGES = {
    'P': (min(X[0, :].min(), newPara[0, :].min()),
          max(X[0, :].max(), newPara[0, :].max())),
    'H': (min(X[1, :].min(), newPara[1, :].min()),
          max(X[1, :].max(), newPara[1, :].max())),
}
dims = ['P', 'H']

# 构造 G* 向量
Gstar = podGvec(X, newPara, PARAM_RANGES, dims)

# 预测 RF
Ypred_new = np.dot(B, Gstar)  # (n_outputs, 1)

# 保存结果（文件名中带参数值，内容只存预测结果）
output_file = os.path.join(
    output_folder,
    f"Y_anyPara_pred_RF_P_{P_val:.6f}_detaH_{detaH_val:.4f}.csv"
)
np.savetxt(output_file, Ypred_new, delimiter=",")

print("Predicted RF at newPara:", newPara.ravel())
print("Saved:", output_file)
