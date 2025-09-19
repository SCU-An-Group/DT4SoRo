import numpy as np
import os
import time

start_time = time.time()

# =========================================================
# 工具函数
# =========================================================

def format_parameter_name(P, H):
    """格式化参数为文件名友好格式"""
    return f"P_{P:.6f}_H_{H:.6f}"


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

    # 有效维度（跨度大于 0）
    valid_mask = (span[:, 0] > 0)
    # 避免除零
    span_safe = np.where(span == 0.0, 1.0, span)

    Xn = (X - min_arr) / span_safe
    Xtestn = (Xtest - min_arr) / span_safe

    # 裁剪和警告
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


def phi_func(r, kernel="gaussian", epsilon=1.0):
    """常见 RBF 核函数"""
    if kernel == "gaussian":
        return np.exp(-(r/epsilon)**2)
    elif kernel == "mq":  # multiquadric
        return np.sqrt(1 + (r/epsilon)**2)
    elif kernel == "imq":  # inverse multiquadric
        return 1.0 / np.sqrt(1 + (r/epsilon)**2)
    elif kernel == "tps":  # thin plate spline
        return r**2 * np.log(r + 1e-12)  # 避免 log(0)
    elif kernel == "linear":  # linear splines
        return r
    elif kernel == "cubic":  # cubic splines
        return r**3
    elif kernel == "sqrt":
        return np.sqrt(r)
    elif kernel == "cqrt":
        return np.cbrt(r)
    else:
        raise ValueError(f"未知核函数: {kernel}")


def rbfBmtx(Y, X, Xtest, kernel="gaussian", epsilon=1.0):
    """
    RBF interpolation coefficient matrix.
    Parameters:
        Y: output matrix (1 × n_train)
        X: parameter matrix (a × n_train)
        Xtest: parameter matrix (a × n_test)
    Returns:
        B, Gstar
    """
    # === 归一化 ===
    PARAM_RANGES = {
        'P': (min(X[0, :].min(), Xtest[0, :].min()),
              max(X[0, :].max(), Xtest[0, :].max())),
        'H': (min(X[1, :].min(), Xtest[1, :].min()),
              max(X[1, :].max(), Xtest[1, :].max())),
    }
    dims = ['P', 'H']

    Xn, Xtestn, valid_mask = normalize_with_ranges(X, Xtest, PARAM_RANGES, dims)

    # 只取有效维度
    Xn_use = Xn[valid_mask, :]
    Xtestn_use = Xtestn[valid_mask, :]

    n = Xn_use.shape[1]  # n_train

    # 构造 G
    G = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            r = np.linalg.norm(Xn_use[:, i] - Xn_use[:, j])
            G[i, j] = phi_func(r, kernel, epsilon)

    # 训练 B
    try:
        B = np.dot(Y, np.linalg.inv(G))
    except np.linalg.LinAlgError:
        B = np.dot(Y, np.linalg.pinv(G))

    # 构造 Gstar
    Gstar = np.zeros((n, Xtestn_use.shape[1]))
    for i in range(Xtestn_use.shape[1]):
        for k in range(n):
            r = np.linalg.norm(Xtestn_use[:, i] - Xn_use[:, k])
            Gstar[k, i] = phi_func(r, kernel, epsilon)

    return B, Gstar


# =========================================================
# 主程序
# =========================================================

def main():
    # === 路径设置 ===
    base_path = r'F:\Fast-pn-actuator\Fpn_bending\Gripper'
    param_file = os.path.join(base_path, "Parameters_P_detaH.py")
    output_folder = os.path.join(base_path, 'output_data')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # === 读取参数 ===
    namespace = {}
    with open(param_file, "r") as file:
        exec(file.read(), namespace)

    X = np.array([namespace["training_P_values"], namespace["training_detaH_values"]])
    Xtest = np.array([namespace["test_P_values"], namespace["test_detaH_values"]])

    # === 读取 RF 数据 ===
    Y_train_file = os.path.join(output_folder, 'Y_training_RF.csv')
    Y_test_file = os.path.join(output_folder, 'Y_test_RF.csv')

    Y_train = np.loadtxt(Y_train_file, delimiter=',')  # shape (1, n_train) 或 (n_train,)
    Y_test = np.loadtxt(Y_test_file, delimiter=',')

    if Y_train.ndim == 1:
        Y_train = Y_train[np.newaxis, :]

    # === 训练 RBF 系数矩阵 & 预测 ===
    kerneltype = 'linear'
    epsilonnum = 0.1
    B, Gstar = rbfBmtx(Y_train, X, Xtest, kernel=kerneltype, epsilon=epsilonnum)

    # 保存 B
    B_file = os.path.join(output_folder, 'B_matrix_RF.csv')
    np.savetxt(B_file, B, delimiter=',', fmt='%f')
    print("B matrix for RF saved to {}".format(B_file))

    # 预测
    Y_pred = np.dot(B, Gstar)

    # 计算误差
    mse = np.mean((Y_pred.flatten() - Y_test.flatten()) ** 2)
    print("Reaction Force Prediction MSE:", mse)

    # 保存预测结果
    for j in range(Xtest.shape[1]):
        P_val = Xtest[0, j]
        H_val = Xtest[1, j]
        param_str = format_parameter_name(P_val, H_val)
        pred_val = Y_pred[0, j]

        out_file = os.path.join(output_folder, f"Y_prediction_RF_{param_str}.csv")
        np.savetxt(out_file, [pred_val], delimiter=',', fmt='%f')
        print(f"Predicted RF for {param_str} saved to {out_file}")

    end_time = time.time()
    print(f"\n总运行时间: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
