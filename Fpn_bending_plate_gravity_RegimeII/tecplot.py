import csv

# 文件名设置（请根据实际文件路径调整）
node_csv_file = 'F:\Fast-pn-actuator\Fpn_bending\Board_new_new\FEModelFiles_test_csv_new_new\Fpn_bending_Board_P_0.031749_detaH_30.7634\Fpn_bending_Board_P_0.031749_detaH_30.7634_last_frame_node.csv'
elem_csv_file = 'F:\Fast-pn-actuator\Fpn_bending\Board_new_new\FEModelFiles_test_csv_new_new\Fpn_bending_Board_P_0.061749_detaH_20.7634\Fpn_bending_Board_P_0.061749_detaH_20.7634_last_frame_elem.csv'

board_node_csv_file = 'F:\Fast-pn-actuator\Fpn_bending\Board_new_new\FEModelFiles_test_csv_new_new\Fpn_bending_Board_P_0.031749_detaH_30.7634\Fpn_bending_Board_P_0.031749_detaH_30.7634_board_nodes.csv'
board_elem_csv_file = 'F:\Fast-pn-actuator\Fpn_bending\Board_new_new\FEModelFiles_test_csv_new_new\Fpn_bending_Board_P_0.031749_detaH_30.7634\Fpn_bending_Board_P_0.031749_detaH_30.7634_board_elements.csv'

output_file = r'F:\Fast-pn-actuator\Fpn_bending\Board_new_new\output_data\tecplot\tecplot_test_P_0.031749_detaH_30.7634.dat'

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
                'mises': row['mises'].strip(),
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
    fout.write('VARIABLES = "X", "Y", "Z", "U", "mises"\n')
    fout.write(f'ZONE T="Zone 1", N={num_nodes}, E={num_elements}, DATAPACKING=POINT, ZONETYPE=FETETRAHEDRON\n')
    
    for node in nodes_sorted:
        try:
            x = float(node['X'] or 0.0)
            y = float(node['Y'] or 0.0)
            z = float(node['Z'] or 0.0)
            u = float(node['U'] or 0.0)
            Smises = float(node['mises'] or 0.0)
            fout.write(f"{x:20.5f}{y:20.5f}{z:20.5f}{u:20.5f}{Smises:20.5f}\n")
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