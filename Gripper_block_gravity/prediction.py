import numpy as np
import os
import time

start_time = time.time()

# =========================================================
# 工具函数
# =========================================================

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

def normalize_with_ranges(X, Xpred, ranges, dims, on_out_of_range='warn+clip'):
    """用指定全局范围归一化；零跨度维度会被忽略"""
    min_arr = np.array([ranges[d][0] for d in dims], dtype=float).reshape(-1, 1)
    max_arr = np.array([ranges[d][1] for d in dims], dtype=float).reshape(-1, 1)
    span = max_arr - min_arr
    valid_mask = (span[:, 0] > 0)
    span_safe = np.where(span == 0.0, 1.0, span)

    Xn = (X - min_arr) / span_safe
    Xpredn = (Xpred - min_arr) / span_safe

    if on_out_of_range in ('clip', 'warn+clip'):
        Xn = np.clip(Xn, 0.0, 1.0)
        Xpredn = np.clip(Xpredn, 0.0, 1.0)

    if on_out_of_range in ('warn', 'warn+clip'):
        def _warn(name, Z):
            if np.any(Z < 0.0) or np.any(Z > 1.0):
                print(f"[WARN] {name} 存在超出全局范围的值（已{('裁剪' if 'clip' in on_out_of_range else '未裁剪')}）。")
        _warn('X_norm', Xn)
        _warn('Xpred_norm', Xpredn)

    if not np.any(valid_mask):
        raise ValueError("所有参数维度的全局跨度均为0，至少需要一个有变化的参数。")

    return Xn, Xpredn, valid_mask

# =========================================================
# 参数设置
# =========================================================

# 新预测参数
newPara = [0.076, 20.0]   # [P, H]

# 定义输出文件夹（保存预测结果以及pod_mode）
base_path = r'F:\Fast-pn-actuator\Fpn_bending\Gripper'
output_folder = os.path.join(base_path, 'output_data')

# 读取参数文件，构造参数矩阵 X 与 Xpred
namespace = {}
with open(base_path + r"/Parameters_P_detaH.py", "r") as file:
    exec(file.read(), namespace)

# 构造二维参数矩阵
X = np.array([namespace["training_P_values"],
              namespace["training_detaH_values"]])  # (2, n_train)
Xpred = np.array(newPara).reshape(2, 1)  # (2, 1)

# ---------------- 策略 A: 指定全局范围 -----------------
PARAM_RANGES = {
    'P': (0.0, 0.09),  # 根据训练集确定范围
    'H': (10.0, 50.0),                  # 手动指定范围（即使全是20.0）
}
dims = ['P', 'H']

# 归一化
X_norm, Xpred_norm, valid_mask = normalize_with_ranges(X, Xpred, PARAM_RANGES, dims)
X_use = X_norm[valid_mask, :]
Xpred_use = Xpred_norm[valid_mask, :]

# 预计算 G, Gstar
G = cdist_cols(X_use)
Gstar = cdist_cols(X_use, Xpred_use)
try:
    G_factor = np.linalg.inv(G)
except np.linalg.LinAlgError:
    G_factor = np.linalg.pinv(G)

# =========================================================
# 预测
# =========================================================

# 用于存放预测结果的字典
predictions = {}
# 定义数据类型列表（注意：这里"X", "Y", "Z"为坐标数据）
data_types = ["X", "Y", "Z", "U", "maxPrincipal", "mises", "F"]

for dtype in data_types:
    # 读取Fa_l以及B矩阵
    Fa_l_file = os.path.join(output_folder, f'Fa_l_{dtype}.csv')
    B_file = os.path.join(output_folder, f'B_matrix_{dtype}.csv')
    
    # 加载数据
    Fa_l = np.loadtxt(Fa_l_file, delimiter=',')
    B = np.loadtxt(B_file, delimiter=',')
    
    # RBF 插值预测
    Ypred = Fa_l @ (B @ Gstar)
    
    # 保存预测结果
    predictions[dtype] = Ypred

# 将同一预测参数的各数据类型结果合并保存
header = ['Node Label'] + data_types
num_nodes = list(predictions.values())[0].shape[0]

P_val, H_val = newPara
param_str = format_parameter_name(P_val, H_val)
combined = np.zeros((num_nodes, 1 + len(data_types)))
combined[:, 0] = np.arange(1, num_nodes + 1)
for i, dtype in enumerate(data_types):
    combined[:, i+1] = predictions[dtype][:, 0]

output_path = os.path.join(output_folder, f'Y_anyPara_pred_all_{param_str}.csv')
np.savetxt(output_path, combined, delimiter=',', fmt='%f',
           header=",".join(header), comments='')
print(f"预测结果已保存: {output_path}")

end_time = time.time()
print(f"\n总运行时间: {end_time - start_time:.2f} 秒")

    















import csv

# 文件名设置（请根据实际文件路径调整）
node_csv_file = output_path
elem_csv_file = 'F:\Fast-pn-actuator\Fpn_bending\Gripper\FEModelFiles_training_csv\Fpn_bending_Board_P_0.09_detaH_20.0\Fpn_bending_Board_P_0.09_detaH_20.0_last_frame_elem.csv'

board_node_csv_file = 'F:\Fast-pn-actuator\Fpn_bending\Gripper\FEModelFiles_training_csv\Fpn_bending_Board_P_0.09_detaH_20.0\Fpn_bending_Board_P_0.09_detaH_20.0_board_nodes.csv'
board_elem_csv_file = 'F:\Fast-pn-actuator\Fpn_bending\Gripper\FEModelFiles_training_csv\Fpn_bending_Board_P_0.09_detaH_20.0\Fpn_bending_Board_P_0.09_detaH_20.0_board_elements.csv'

output_file = r'F:\Fast-pn-actuator\Fpn_bending\Gripper\output_data\tecplot\tecplot_pred_P_0.076_H_20.0.dat'

# 读取节点数据
nodes = []
with open(node_csv_file, 'r', newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            node_label = float(row['Node Label'].strip())
            nodes.append({
                'Node Label': node_label,
                'X': row['X'].strip(),
                'Y': row['Y'].strip(),
                'Z': row['Z'].strip(),
                'U': row['U'].strip(),
                'maxPrincipal': row['maxPrincipal'].strip(),
                'mises': row['mises'].strip(),
                'F': row['F'].strip()
            })
        except ValueError as e:
            print(f"节点标签转换错误：{row['Node Label']}，错误信息：{e}")
            continue



# 按 Node Label 升序排序
nodes_sorted = sorted(nodes, key=lambda r: int(r['Node Label']))
num_nodes = len(nodes_sorted)

# 创建节点标签到Tecplot节点编号的映射
node_label_to_tec_id = {node['Node Label']: idx + 1 for idx, node in enumerate(nodes_sorted)}

# 读取单元数据，提取所有节点编号
elements = []
with open(elem_csv_file, 'r', newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        connectivity = []
        # 假设表头中所有以 "Node " 开头的字段均为节点编号
        for key in row:
            if key.startswith("Node "):
                val = row[key].strip()
                if val:  # 如果该字段非空，则添加
                    connectivity.append(val)
        
        # 检查是否为C3D10H单元（10节点）
        if len(connectivity) == 10:
            # # 转换为Tecplot节点编号并取前4个
            # tec_connectivity = [node_label_to_tec_id.get(n) for n in connectivity[:4]]
            # elements.append({
            #     'Element Label': row['Element Label'].strip(),
            #     'connectivity': tec_connectivity
            tec_connectivity = []
            for n in connectivity[:4]:
                n = float(n)
                if n in node_label_to_tec_id:
                    tec_connectivity.append(node_label_to_tec_id[n])
                else:
                    print(f"警告：节点编号 {n} 未找到对应的Tecplot编号，跳过该单元")
                    break  # 跳过该单元
            else:
                # 如果没有触发 break，则添加该单元
                elements.append({
                    'Element Label': row['Element Label'].strip(),
                    'connectivity': tec_connectivity
            })

num_elements = len(elements)

# 生成 Tecplot ASCII 文件
with open(output_file, 'w', newline='') as fout:
    fout.write('TITLE = "FE Mesh (XYZ only)"\n')
    fout.write('VARIABLES = "X", "Y", "Z", "U", "maxPrincipal", "mises", "F"\n')
    fout.write(f'ZONE T="Zone 1", N={num_nodes}, E={num_elements}, DATAPACKING=POINT, ZONETYPE=FETETRAHEDRON\n')
    
    for node in nodes_sorted:
        try:
            x = float(node['X'] or 0.0)
            y = float(node['Y'] or 0.0)
            z = float(node['Z'] or 0.0)
            u = float(node['U'] or 0.0)
            SmaxP = float(node['maxPrincipal'] or 0.0)
            Smises = float(node['mises'] or 0.0)
            f = float(node['F'] or 0.0)
            fout.write(f"{x:20.5f}{y:20.5f}{z:20.5f}{u:20.5f}{SmaxP:20.5f}{Smises:20.5f}{f:20.5f}\n")
        except ValueError as e:
            print(f"节点数据转换错误，节点编号: {node['Node Label']}，错误: {e}")
            continue
    
    for elem in elements:
        conn_str = ' '.join(f"{n:10d}" for n in elem['connectivity'])
        fout.write(conn_str + '\n')

print(f"第一实例 Tecplot 文件已生成：{output_file}，包含 {num_elements} 个四面体单元")


# 读取板件节点数据（只取 X, Y, Z）
board_nodes = []
with open(board_node_csv_file, 'r', newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            x = float(row['X'] or 0.0)
            y = float(row['Y'] or 0.0)
            z = float(row['Z'] or 0.0)
            board_nodes.append({'X': x, 'Y': y, 'Z': z})
        except ValueError as e:
            print(f"板节点数据转换错误：{row} 错误信息: {e}")
            continue

num_board_nodes = len(board_nodes)

# 读取板件网格数据，读取前 8 个节点的节点编号
board_elements = []
with open(board_elem_csv_file, 'r', newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        connectivity = []
        # 假设所有以 "Node " 开头的字段为节点编号
        for key in row:
            if key.startswith("Node "):
                val = row[key].strip()
                if val:
                    connectivity.append(val)
        if len(connectivity) >= 8:
            try:
                # 只取前 8 个节点，并转换为整数（假设节点编号为数值型）
                conn = [int(float(n)) for n in connectivity[:8]]
                board_elements.append({
                    'Element Label': row.get('Element Label', '').strip(),
                    'connectivity': conn
                })
            except ValueError as e:
                print(f"板单元数据转换错误: {e}")
                continue

num_board_elements = len(board_elements)

# 写入第二实例信息到 Tecplot 文件（追加到原文件末尾）
with open(output_file, 'a', newline='') as fout:
    fout.write('\n')  # 分隔符
    # 只写 Zone 头，沿用全局的 VARIABLES 定义（共6列数据）
    fout.write(f'ZONE T="Zone 2", N={num_board_nodes}, E={num_board_elements}, DATAPACKING=POINT, ZONETYPE=FEBRICK\n')

    # 写入板件节点数据
    for node in board_nodes:
        fout.write(f"{node['X']:20.5f}{node['Y']:20.5f}{node['Z']:20.5f}{0.0:20.5f}{0.0:20.5f}{0.0:20.5f}{0.0:20.5f}\n")

    # 写入板件单元连接信息（仅前 8 个节点）
    for elem in board_elements:
        conn_str = ' '.join(f"{n:10d}" for n in elem['connectivity'])
        fout.write(conn_str + '\n')

print(f"Tecplot 文件已生成：{output_file}，包含 {num_elements} 个四面体单元和 {num_board_elements} 个八面体单元")

end_time = time.time()
print(f"\n总运行时间: {end_time - start_time:.2f} 秒")