import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from skmultilearn.model_selection import iterative_train_test_split

# ==============================================================================
# 1. 配置区 - 请根据您的项目修改这里
# ==============================================================================

# 增强后数据的源路径
SOURCE_IMAGE_DIR = 'dataset/images'
SOURCE_LABEL_DIR = 'dataset/labels'

# 分割后数据的保存路径
OUTPUT_DIR = 'dataset/split'

# 您的类别名称，顺序必须与 class_id 对应
CLASS_NAMES = [
    'hanyan', 'zhadian', 'handon', 'hangao', 'hanzha', 
    'duanhan', 'hanchuan', 'hanpian'
]

# 定义训练集和验证集的比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2 # 确保 TRAIN_RATIO + VAL_RATIO = 1.0


# ==============================================================================
# 2. 修正后的辅助函数
# ==============================================================================

def analyze_distribution(label_dir, file_list, class_names):
    """分析给定文件列表中的类别分布"""
    num_classes = len(class_names)
    distribution = np.zeros(num_classes, dtype=int)
    
    for filename in file_list:
        label_path = os.path.join(label_dir, filename)
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(float(parts[0]))
                        if 0 <= class_id < num_classes:
                            distribution[class_id] += 1
    
    print("\n类别分布情况:")
    print("=" * 30)
    df = pd.DataFrame({
        '类别名称': class_names,
        '实例数量': distribution
    })
    print(df)
    print("=" * 30)
    return distribution


# <<< 核心修正点在这里 >>>
def copy_files(file_list, image_src, label_src, image_dest, label_dest):
    """
    根据标签文件列表，查找并复制对应的标签和图片文件。
    这个版本会自动查找图片的后缀名。
    """
    os.makedirs(image_dest, exist_ok=True)
    os.makedirs(label_dest, exist_ok=True)
    
    # 定义可能的图片文件后缀
    possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    
    for filename_txt in tqdm(file_list, desc=f"复制到 {os.path.basename(os.path.dirname(image_dest))}"):
        # 1. 复制标签文件 (这个很简单)
        label_path_src = os.path.join(label_src, filename_txt)
        if os.path.exists(label_path_src):
            shutil.copy(label_path_src, os.path.join(label_dest, filename_txt))

        # 2. 查找并复制对应的图片文件
        base_name = os.path.splitext(filename_txt)[0]
        image_found = False
        for ext in possible_extensions:
            # 构造可能的图片文件名 (包括大小写)
            image_name_lower = base_name + ext.lower()
            image_name_upper = base_name + ext.upper()
            
            image_path_src_lower = os.path.join(image_src, image_name_lower)
            image_path_src_upper = os.path.join(image_src, image_name_upper)

            # 检查小写后缀的图片是否存在
            if os.path.exists(image_path_src_lower):
                shutil.copy(image_path_src_lower, os.path.join(image_dest, image_name_lower))
                image_found = True
                break # 找到了就跳出循环
            
            # 检查大写后缀的图片是否存在
            if os.path.exists(image_path_src_upper):
                shutil.copy(image_path_src_upper, os.path.join(image_dest, image_name_upper))
                image_found = True
                break # 找到了就跳出循环

        # 如果循环结束了还没找到对应的图片，打印一个警告
        if not image_found:
            print(f"警告：标签文件 '{filename_txt}' 存在，但没有找到对应的图片文件。")

# ==============================================================================
# 3. 主执行脚本 (无需改动)
# ==============================================================================

def main():
    print("开始进行多标签分层抽样...")
    
    label_files = [f for f in os.listdir(SOURCE_LABEL_DIR) if f.endswith('.txt')]
    num_samples = len(label_files)
    num_classes = len(CLASS_NAMES)

    print("正在分析所有文件的类别信息...")
    Y = np.zeros((num_samples, num_classes), dtype=int)
    for i, filename in enumerate(tqdm(label_files, desc="分析标签")):
        label_path = os.path.join(SOURCE_LABEL_DIR, filename)
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        class_id = int(float(parts[0]))
                        if 0 <= class_id < num_classes:
                            Y[i, class_id] = 1
                    except ValueError:
                        print(f"警告：在文件 {filename} 中发现无法转换的行，已跳过: {line.strip()}")

    X = np.arange(num_samples).reshape(-1, 1)

    print("\n正在执行分层抽样...")
    train_indices, _, val_indices, _ = iterative_train_test_split(X, Y, test_size=VAL_RATIO)
    
    train_files = [label_files[i] for i in train_indices.flatten()]
    val_files = [label_files[i] for i in val_indices.flatten()]

    print(f"抽样完成. 训练集样本数: {len(train_files)}, 验证集样本数: {len(val_files)}")
    
    train_image_dest = os.path.join(OUTPUT_DIR, 'train', 'images')
    train_label_dest = os.path.join(OUTPUT_DIR, 'train', 'labels')
    val_image_dest = os.path.join(OUTPUT_DIR, 'val', 'images')
    val_label_dest = os.path.join(OUTPUT_DIR, 'val', 'labels')

    copy_files(train_files, SOURCE_IMAGE_DIR, SOURCE_LABEL_DIR, train_image_dest, train_label_dest)
    copy_files(val_files, SOURCE_IMAGE_DIR, SOURCE_LABEL_DIR, val_image_dest, val_label_dest)
    
    print("\n--- 训练集 ---")
    analyze_distribution(train_label_dest, os.listdir(train_label_dest), CLASS_NAMES)
    
    print("\n--- 验证集 ---")
    analyze_distribution(val_label_dest, os.listdir(val_label_dest), CLASS_NAMES)

    print("\n数据集分割完成!")
    print(f"数据已保存到: {os.path.abspath(OUTPUT_DIR)}")
    print("现在您可以使用这个 'split_dataset' 文件夹来配置您的YOLO训练文件了。")

if __name__ == '__main__':
    main()