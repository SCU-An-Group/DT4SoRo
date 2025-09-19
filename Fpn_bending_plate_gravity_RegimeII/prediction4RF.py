# -*- coding: utf-8 -*-
"""
Predict RF at a single new parameter using pre-trained RBF coefficients
"""

import numpy as np
import os

# ========== RBF 方法 ==========
def podGvec(X, Xtest):
    """Construct G* vector for new points."""
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

# ========== 主程序 ==========
base_path = r'F:\Fast-pn-actuator\Fpn_bending\GitHub_code\Fpn_bending_plate_gravity_RegimeII'
param_file = os.path.join(base_path, "Parameters_P_detaH.py")
output_folder = os.path.join(base_path, 'output_data')

# 读取参数
namespace = {}
with open( param_file ) as f:
    exec(f.read(), namespace)

X = np.array([namespace["training_P_values"],
              namespace["training_detaH_values"]])  # (2, n_train)

# 读取 B 矩阵
B = np.loadtxt(os.path.join(output_folder, "B_matrix_RF.csv"), delimiter=",")

# 定义新的参数点
newPara = np.array([[0.070469], [29.744473]])  # shape (2,1)
P_val, detaH_val = newPara.ravel()

# 构造 G* 向量
Gstar = podGvec(X, newPara)

# 预测 RF
Ypred_new = np.dot(B, Gstar)  # (n_outputs, 1)

# 保存结果（文件名中带参数值，内容只存预测结果）
output_file = os.path.join(
    output_folder,
    f"Y_prediction_RF_anyPara_P_{P_val:.6f}_detaH_{detaH_val:.4f}.csv"
)
np.savetxt(output_file, Ypred_new, delimiter=",")

print("Predicted RF at newPara:", newPara.ravel())
print("Saved:", output_file)