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
            "P_values",
            "test_P_values"
        ]
        
        for param in required_params:
            if param not in namespace:
                raise ValueError(f"Missing required parameter: {param}")
            
        return namespace
        
    except Exception as e:
        logging.error(f"参数加载失败: {str(e)}")
        raise

def build_file_paths(base_dir, P_values, set_type):
    """构建文件路径并验证存在性"""
    paths = []
    for P in P_values:
        folder_name = f"Fpn_bending_Gravity_P_{P}"
        folder_path = Path(base_dir) / f"FEModelFiles_{set_type}_csv" / folder_name
        
        file_name = f"Fpn_bending_Gravity_P_{P}_last_frame_node.csv"
        file_path = folder_path / file_name
        
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
                
            # 按特征存储数据
            for col in data_dict:
                data_dict[col].append(df[col].values)
            
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
        
        sample_counts = [len(col) for col in data_columns]
        if len(set(sample_counts)) > 1:
            logging.error(f"特征 {feature} 数据长度不一致: {sample_counts}")
            raise ValueError(f"特征 {feature} 数据长度不一致")
        
        df = pd.DataFrame(data_columns).T
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

def main():
    try:
        base_path = Path(r"F:/Fast-pn-actuator/Fpn_bending/GitHub_code/Fpn_bending_noplate_gravity_RegimeI")
        params = load_parameters(base_path/"Parameters_P.py")
        output_dir = base_path / "output_data"
        
        # 处理训练集
        train_paths = build_file_paths(base_path, params["P_values"], "training")
        train_data, train_merged = process_dataset(train_paths, "training")
        save_dataset(train_data, output_dir, "training")
        save_merged_dataset(train_merged, output_dir, "training")
        
        # 处理测试集
        test_paths = build_file_paths(base_path, params["test_P_values"], "test")
        test_data, test_merged = process_dataset(test_paths, "test")
        save_dataset(test_data, output_dir, "test")
        save_merged_dataset(test_merged, output_dir, "test")
        
        logging.info("数据处理流程完成")
        
    except Exception as e:
        logging.error(f"主流程异常: {str(e)}")
        raise

if __name__ == "__main__":
    main()