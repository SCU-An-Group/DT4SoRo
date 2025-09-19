import csv

node_csv_file = r'F:\Fast-pn-actuator\Fpn_bending\Board_new_new\FEModelFiles_test_csv_new_new\Fpn_bending_Board_P_0.031749_detaH_30.7634\Fpn_bending_Board_P_0.031749_detaH_30.7634_last_frame_node.csv'
pred_csv_file = r'F:\Fast-pn-actuator\Fpn_bending\Board_new_new\output_data\Y_anyPara_pred_all_P_0.031749_H_30.7634.csv'
error_file = r'F:\Fast-pn-actuator\Fpn_bending\Board_new_new\output_data\error_P_0.031749_detaH_30.7634.csv'

# 读取 node.csv 并存储数据
node_data = {}
with open(node_csv_file, 'r', newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        node_label = int(float(row['Node Label']))
        node_data[node_label] = {
            'X': row['X'].strip(),
            'Y': row['Y'].strip(),
            'Z': row['Z'].strip(),
            'U': row['U'].strip(),
            'mises': row['mises'].strip(),
        }

# 读取 prediction.csv 并存储数据
prediction_data = {}
with open(pred_csv_file, 'r', newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        node_label = int(float(row['Node Label']))
        prediction_data[node_label] = {
            'U': row['U'].strip(),
            'mises': row['mises'].strip()
        }

# 生成 error.csv
with open(error_file, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['Node Label', 'X', 'Y', 'Z', 'U', 'mises'])
    
    count_written = 0
    for node_label, node_info in node_data.items():
        if node_label in prediction_data:
            pred_info = prediction_data[node_label]
            try:
                u_error = abs(float(node_info['U']) - float(pred_info['U']))
                mises_error = abs(float(node_info['mises']) - float(pred_info['mises']))
            except Exception as e:
                print("数值转换错误，节点:", node_label, e)
                continue
            writer.writerow([
                int(node_label),
                node_info['X'],
                node_info['Y'],
                node_info['Z'],
                u_error,
                mises_error,
            ])
            count_written += 1
        else:
            print("节点在 prediction 中未匹配到:", repr(node_label))
            
print("error.csv 文件已生成")
