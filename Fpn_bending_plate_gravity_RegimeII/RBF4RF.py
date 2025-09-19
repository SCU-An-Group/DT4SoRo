import numpy as np
import os
import time

start_time = time.time()

def rbfBmtx(Y, X, kernel="gaussian", epsilon=1.0):
    """
    RBF interpolation coefficient matrix.
    Parameters:
        Y: output matrix (1 × n_train)
        X: parameter matrix (a × n_train)
        kernel: type of RBF kernel ("gaussian", "mq", "imq", "tps")
        epsilon: shape parameter for RBF
    Returns:
        B: RBF interpolation coefficient matrix
    """
    # Normalize parameters to [0, 1]
    x = np.zeros_like(X, dtype=float)
    minX = np.min(X, axis=1)
    maxX = np.max(X, axis=1)

    for j in range(X.shape[0]):
        for i in range(X.shape[1]):
            x[j, i] = (X[j, i] - minX[j]) / (maxX[j] - minX[j])

    n = x.shape[1]  # data_size
    G = np.zeros((n, n), dtype=float)

    def phi(r):
        if kernel == "gaussian":
            return np.exp(-(r/epsilon)**2)
        elif kernel == "mq":  # multiquadric
            return np.sqrt(1 + (r/epsilon)**2)
        elif kernel == "imq":  # inverse multiquadric
            return 1.0 / np.sqrt(1 + (r/epsilon)**2)
        elif kernel == "tps":  # thin plate spline
            return r**2 * np.log(r + 1e-12)  # avoid log(0)
        elif kernel == "linear":  # linearsplines
            return r
        elif kernel == "cubic":  # cubicsplines
            return r**3
        elif kernel == "sqrt":
            return np.sqrt(r)
        elif kernel == "cqrt":
            return np.cbrt(r)
        
    # Construct G matrix
    for i in range(n):
        for j in range(n):
            r = np.linalg.norm(x[:, i] - x[:, j])
            G[i, j] = phi(r)

    # Solve for B
    try:
        B = np.dot(Y, np.linalg.inv(G))
    except np.linalg.LinAlgError:
        B = np.dot(Y, np.linalg.pinv(G))
    return B, minX, maxX  # 把归一化参数也返回


def rbfGvec(X, Xtest, minX, maxX, kernel="gaussian", epsilon=1.0):
    """
    Construct Gstar vector for given test parameters.
    """
    a, n = X.shape
    x = np.zeros_like(X, dtype=float)
    normalized_Xtest = np.zeros_like(Xtest, dtype=float)

    for j in range(a):
        for i in range(n):
            x[j, i] = (X[j, i] - minX[j]) / (maxX[j] - minX[j])
    for j in range(Xtest.shape[0]):
        for i in range(Xtest.shape[1]):
            normalized_Xtest[j, i] = (Xtest[j, i] - minX[j]) / (maxX[j] - minX[j])

    def phi(r):
        if kernel == "gaussian":
            return np.exp(-(r/epsilon)**2)
        elif kernel == "mq":  # multiquadric
            return np.sqrt(1 + (r/epsilon)**2)
        elif kernel == "imq":  # inverse multiquadric
            return 1.0 / np.sqrt(1 + (r/epsilon)**2)
        elif kernel == "tps":  # thin plate spline
            return r**2 * np.log(r + 1e-12)  # avoid log(0)
        elif kernel == "linear":  # linearsplines
            return r
        elif kernel == "cubic":  # cubicsplines
            return r**3
        elif kernel == "sqrt":
            return np.sqrt(r)
        elif kernel == "cqrt":
            return np.cbrt(r)        

    Gstar = np.zeros((n, Xtest.shape[1]))
    for i in range(Xtest.shape[1]):
        for k in range(n):
            r = np.linalg.norm(normalized_Xtest[:, i] - x[:, k])
            Gstar[k, i] = phi(r)
    return Gstar


def format_parameter_name(P, H):
    """格式化参数为文件名友好格式"""
    return f"P_{P:.6f}_H_{H:.6f}"



def main():
    # === 路径设置 ===
    base_path = r'F:\Fast-pn-actuator\Fpn_bending\GitHub_code\Fpn_bending_plate_gravity_RegimeII'
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

    # 确保 Y_train 是 (1, n_train) 形式
    if Y_train.ndim == 1:
        Y_train = Y_train[np.newaxis, :]

    # === 训练 RBF 系数矩阵 ===
    # B = rbfBmtx(Y_train, X)
    kerneltype = 'linear'
    epsilonnum = 0.1
    B, minX, maxX = rbfBmtx(Y_train, X, kernel = kerneltype, epsilon = epsilonnum)
    B_file_template = os.path.join(output_folder, 'B_matrix_RF.csv')
    
    # 立即保存B矩阵到文件
    current_B_file = B_file_template
    np.savetxt(current_B_file, B, delimiter=',', fmt='%f')
    print("B matrix for RF saved to {}".format(current_B_file))
    
    # === 对测试集预测 ===
    # Gstar = rbfGvec(X, Xtest)
    Gstar = rbfGvec(X, Xtest, minX, maxX, kernel = kerneltype, epsilon = epsilonnum)
    Y_pred = np.dot(B, Gstar)  # shape (1, n_test)

    # === 计算误差 ===
    mse = np.mean((Y_pred.flatten() - Y_test.flatten()) ** 2)
    print("Reaction Force Prediction MSE:", mse)

    # === 保存预测结果 ===
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
