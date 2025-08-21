# --- START OF FILE traincv_enhanced.py ---

from ultralytics import YOLO
import torch
import os
import yaml
from sklearn.model_selection import KFold
import shutil
import glob

def train_yolov11_with_cv(
    data_yaml_path, 
    model_variant='yolov11n.pt', 
    epochs=100, 
    imgsz=640, 
    batch_size=16, 
    device='cpu', 
    n_splits=5,
    project_name='YOLOv11_CV_Results'
):
    """
    一个用于通过五折交叉验证训练YOLOv11模型的主函数。
    此版本集成了Mosaic, Mixup, CopyPaste等高级数据增强。

    参数:
    - data_yaml_path (str): 指向您的原始数据集 .yaml 文件的路径。
    - model_variant (str): 要使用的YOLOv11模型变体。
    - epochs (int): 每一折的训练轮数。
    - imgsz (int): 训练时输入的图像尺寸。
    - batch_size (int): 每个批次的图像数量。
    - device (str): 训练设备, 'cpu' 或 '0' (对于第一个GPU)。
    - n_splits (int): 交叉验证的折数。
    - project_name (str): 保存所有交叉验证结果的总项目文件夹名称。
    """
    temp_files = [] # 用于记录临时文件，以便最后清理

    try:
        # 检查是否有可用的GPU
        if device == '0' and not torch.cuda.is_available():
            print("警告: 请求使用GPU，但未检测到CUDA。将自动切换到CPU。")
            device = 'cpu'

        # 1. 加载原始的 data.yaml 文件
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)

        # 获取训练图片的绝对路径
        base_path = os.path.dirname(data_yaml_path)
        train_images_path = os.path.join(base_path, data_config['train'])
        
        # 获取所有图片文件列表（支持多种格式）
        image_extensions = ['.jpg', '.jpeg', '.png']
        all_images = []
        for ext in image_extensions:
            all_images.extend(glob.glob(os.path.join(train_images_path, f'*{ext}')))
        
        all_images = sorted(list(set(all_images))) # 去重并排序

        if not all_images:
            raise FileNotFoundError(f"在目录 '{train_images_path}' 中未找到任何图片文件。请检查您的 dianchi.yaml 文件中的 'train' 路径。")

        # 2. 初始化K-Fold交叉验证
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42) # random_state保证可复现

        # 3. 开始交叉验证循环
        for fold, (train_indices, val_indices) in enumerate(kf.split(all_images)):
            print(f"\n{'='*20} 开始第 {fold+1}/{n_splits} 折交叉验证 {'='*20}")
            
            # 创建当前折的训练集和验证集列表
            train_files = [all_images[i] for i in train_indices]
            val_files = [all_images[i] for i in val_indices]
            
            # 4. 创建一个临时的YAML文件用于当前折的训练
            fold_yaml_path = os.path.join(base_path, f'dianchi_fold_{fold+1}.yaml')
            
            # 动态生成训练和验证集的txt文件
            train_txt_path = os.path.join(base_path, f'train_fold_{fold+1}.txt')
            val_txt_path = os.path.join(base_path, f'val_fold_{fold+1}.txt')
            temp_files.extend([fold_yaml_path, train_txt_path, val_txt_path])

            with open(train_txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(train_files))
            
            with open(val_txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(val_files))

            # 更新data_config以指向新的文件列表
            fold_data_config = data_config.copy()
            fold_data_config['train'] = train_txt_path
            fold_data_config['val'] = val_txt_path
            
            with open(fold_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(fold_data_config, f, allow_unicode=True)
            
            # 5. 训练模型
            model = YOLO(model_variant)
            
            # 为每一折的训练创建一个单独的输出目录
            run_name = f'fold_{fold+1}'

            print(f"--- 开始训练第 {fold+1} 折 ---")
            model.train(
                # --- 基础配置 ---
                data=fold_yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                device=device,
                project=project_name,
                name=run_name,
                plots=True,
                
                # --- 高级数据增强配置 ---
                mosaic=1.0,      # 启用Mosaic增强 (强烈推荐)
                mixup=0.1,       # 启用Mixup增强
                copy_paste=0.1,  # 启用CopyPaste增强 (功能类似CutMix)
                
                # --- 其他建议的超参数 ---
                patience=20,     # 如果20个epoch没有性能提升，则提前停止
                optimizer='AdamW', # AdamW优化器通常表现很好
                # resume=False # 在交叉验证中，每一折都应从头开始训练
            )
            
            print(f"--- 第 {fold+1}/{n_splits} 折训练完成 ---")

        print(f"\n{'='*20} 所有折的交叉验证训练全部完成！ {'='*20}")
        print(f"所有结果保存在项目文件夹: '{project_name}'")

    except Exception as e:
        print(f"训练过程中发生错误: {e}")
    finally:
        # 6. 清理生成的临时文件
        print("\n正在清理临时文件...")
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"已删除: {file_path}")
        print("清理完成。")


if __name__ == '__main__':
    
    # 指向您配置好的 .yaml 文件
    DATA_YAML_PATH = 'dianchi.yaml'

    # 选择一个YOLOv11的模型变体
    MODEL_VARIANT = 'yolo11x.pt' # 确保您有这个预训练权重文件

    # 训练超参数
    EPOCHS = 10       # 交叉验证时，每折的epoch可以适当减少
    IMAGE_SIZE = 640
    BATCH_SIZE = 8    # 根据您的GPU显存进行调整
    DEVICE = '0'      # 使用第一个GPU。如果没有GPU，请设置为 'cpu'
    N_SPLITS = 5      # 折数
    PROJECT_NAME = 'YOLOv11_Dianchi_CV_Results' # 定义一个清晰的项目名称

    # --- 开始带有高级增强的交叉验证训练 ---
    train_yolov11_with_cv(
        data_yaml_path=DATA_YAML_PATH,
        model_variant=MODEL_VARIANT,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        n_splits=N_SPLITS,
        project_name=PROJECT_NAME
    )