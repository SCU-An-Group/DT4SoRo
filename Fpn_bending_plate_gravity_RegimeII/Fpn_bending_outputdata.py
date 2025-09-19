import numpy as np
import os
import pandas as pd
from pathlib import Path
import logging

# 配置日志系统
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_parameters(param_file):
    """安全加载参数文件"""
    namespace = {}
    try:
        with open(param_file, "r") as f:
            exec(f.read(), namespace)
        
        required_params = [
            "training_P_values", "training_detaH_values",
            "test_P_values", "test_detaH_values"
        ]
        
        for param in required_params:
            if param not in namespace:
                raise ValueError(f"Missing required parameter: {param}")
                
        # 验证参数长度一致性
        if len(namespace["training_P_values"]) != len(namespace["training_detaH_values"]):
            raise ValueError("Training parameters length mismatch")
            
        if len(namespace["test_P_values"]) != len(namespace["test_detaH_values"]):
            raise ValueError("Test parameters length mismatch")
            
        return namespace
        
    except Exception as e:
        logging.error(f"参数加载失败: {str(e)}")
        raise

def build_file_paths(base_dir, P_values, detaH_values, set_type):
    """构建文件路径并验证存在性"""
    paths = []
    for P, detaH in zip(P_values, detaH_values):
        # 构建文件夹路径
        folder_name = f"Fpn_bending_Board_P_{P}_detaH_{detaH}"
        folder_path = Path(base_dir) / f"FEModelFiles_{set_type}_csv_new_new" / folder_name
        
        # 构建文件路径
        file_name = f"Fpn_bending_Board_P_{P}_detaH_{detaH}_last_frame_node.csv"
        file_path = folder_path / file_name
        
        # 验证路径有效性
        if not file_path.exists():
            logging.warning(f"文件不存在: {file_path}")
            continue
            
        paths.append(file_path)
        
    return paths

def process_dataset(file_paths, dataset_type):
    """处理数据集并返回结构化数据"""
    data_dict = {
        'X': [], 'Y': [], 'Z': [],
        'U': [], 'mises': []
    }
    merged_columns = []  # 存储合并后的数据列
     
    for idx, fp in enumerate(file_paths):
        try:
            df = pd.read_csv(fp)
            
            # 验证必要列存在
            required_cols = list(data_dict.keys())
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"缺少必要列: {missing}")
                
            # 按特征存储数据（每个参数的整列数据）
            for col in data_dict:
                data_dict[col].append(df[col].values)  # 直接存储numpy数组

            # 新增：创建合并数据列（按行展开）
            merged_data = df[required_cols].values.flatten()  # 展平为一维数组
            merged_columns.append(merged_data)
            
            logging.info(f"已处理 {dataset_type} 样本 {idx+1}/{len(file_paths)} | 节点数 {len(df)}")
            
        except Exception as e:
            logging.error(f"处理文件 {fp} 失败: {str(e)}")
            continue
            
    return data_dict, merged_columns

def save_dataset(data_dict, output_dir, dataset_type):
    """保存数据集到指定目录（按特征类型分文件）"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for feature, data_columns in data_dict.items():
        if not data_columns:
            logging.warning(f"无有效 {feature} 数据可保存")
            continue
        
        # 检查数据维度一致性
        sample_counts = [len(col) for col in data_columns]
        if len(set(sample_counts)) > 1:
            logging.error(f"特征 {feature} 数据长度不一致: {sample_counts}")
            raise ValueError(f"特征 {feature} 数据长度不一致")
        
        # 创建DataFrame（每列对应一个参数）
        df = pd.DataFrame(data_columns).T
        
        # 保存文件（无表头）
        filename = output_dir / f"Y_{dataset_type}_{feature}.csv"
        df.to_csv(filename, index=False, header=False)
        logging.info(f"已保存 {filename} | 维度 {df.shape}")

def save_merged_dataset(merged_data, output_dir, dataset_type):
    """保存合并后的完整数据集"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not merged_data:
        logging.warning(f"无合并数据可保存: {dataset_type}")
        return
    
    # 转换为DataFrame并转置（每列对应一个参数）
    df = pd.DataFrame(merged_data).T
    filename = output_dir / f"Y_{dataset_type}.csv"
    df.to_csv(filename, index=False, header=False)
    logging.info(f"已保存合并数据集 {filename} | 维度 {df.shape}")


def process_reaction_forces(base_dir, P_values, detaH_values, set_type, output_dir):
    """提取每个参数对应的 reaction force 文件的最后一个RF值"""
    rf_values = []
    
    for P, detaH in zip(P_values, detaH_values):
        folder_name = f"Fpn_bending_Board_P_{P}_detaH_{detaH}"
        folder_path = Path(base_dir) / f"FEModelFiles_{set_type}_csv_new_new" / folder_name
        file_name = f"Fpn_bending_Board_P_{P}_detaH_{detaH}_reaction_forces.csv"
        file_path = folder_path / file_name
        
        if not file_path.exists():
            logging.warning(f"反力文件不存在: {file_path}")
            continue
        
        try:
            df = pd.read_csv(file_path)
            if df.shape[1] < 2:
                raise ValueError("反力文件列数不足，至少需要2列: [time, RF]")
            
            # 提取最后一个 RF 值
            last_rf = df.iloc[-1, 1]  # 第二列（RF）
            rf_values.append(last_rf)
            logging.info(f"已提取 {set_type} P={P}, detaH={detaH} 的最后RF值: {last_rf}")
        
        except Exception as e:
            logging.error(f"读取反力文件失败 {file_path}: {str(e)}")
            continue
    
    # 保存结果
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if rf_values:
        df_out = pd.DataFrame([rf_values])  # 一行，每列一个参数
        filename = output_dir / f"Y_{set_type}_RF.csv"
        df_out.to_csv(filename, index=False, header=False)
        logging.info(f"已保存 {filename} | 维度 {df_out.shape}")
    else:
        logging.warning(f"未提取到任何 {set_type} RF 值")

def main():
    try:
        # 参数配置
        base_path = Path(r"F:/Fast-pn-actuator/Fpn_bending/GitHub_code/Fpn_bending_plate_gravity_RegimeII")
        params = load_parameters(base_path/"Parameters_P_detaH.py")
        output_dir = base_path / "output_data"
        
        # 处理训练集
        train_paths = build_file_paths(
            base_path, 
            params["training_P_values"],
            params["training_detaH_values"],
            "training"
        )
        train_data, train_merged = process_dataset(train_paths, "training")
        save_dataset(train_data, output_dir, "training")
        save_merged_dataset(train_merged, output_dir, "training")

        # 处理测试集
        test_paths = build_file_paths(
            base_path,
            params["test_P_values"],
            params["test_detaH_values"],
            "test"
        )
        test_data, test_merged = process_dataset(test_paths, "test")
        save_dataset(test_data, output_dir, "test")
        save_merged_dataset(test_merged, output_dir, "test")
        
        logging.info("数据处理流程完成")
        
        process_reaction_forces(
        base_path,
        params["training_P_values"],
        params["training_detaH_values"],
        "training",
        output_dir
        )

        process_reaction_forces(
        base_path,
        params["test_P_values"],
        params["test_detaH_values"],
        "test",
        output_dir
        )
        
        
        
        
    except Exception as e:
        logging.error(f"主流程异常: {str(e)}")
        raise

if __name__ == "__main__":
    main()