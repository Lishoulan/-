# --- START OF FILE trainmain_weak_aug.py ---

from ultralytics import YOLO
import torch

def train_yolov11_weak_aug(data_yaml_path, model_variant='yolov11n.pt', epochs=100, imgsz=640, batch_size=16, device='cpu'):
    """
    一个用于训练YOLOv11模型的主函数，配置了“弱数据增强”。

    参数:
    - data_yaml_path (str): 指向您的数据集 .yaml 文件的路径。
    - model_variant (str): 要使用的YOLOv11模型变体。
    - epochs (int): 训练的总轮数。
    - imgsz (int): 训练时输入的图像尺寸。
    - batch_size (int): 每个批次的图像数量。
    - device (str): 训练设备, 'cpu' 或 '0' (对于第一个GPU)。
    """
    try:
        # 检查是否有可用的GPU
        if device == '0' and not torch.cuda.is_available():
            print("警告: 请求使用GPU，但未检测到CUDA。将自动切换到CPU。")
            device = 'cpu'
            
        model = YOLO(model_variant)

        # 开始训练模型
        # 结果将保存在 'runs/segment/train' 或 'runs/detect/train' 目录下
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            plots=True,
            resume=False
        )
        
        print("训练完成！最佳模型保存在:", results.save_dir / 'weights' / 'best.pt')
    except Exception as e:
        print(f"训练过程中发生错误: {e}")

if __name__ == '__main__':
    # 1. 确保您的 dianchi.yaml 文件已正确配置训练集和验证集路径
    DATA_YAML_PATH = 'dianchi.yaml'

    # 2. 选择一个YOLOv11的模型变体
    MODEL_VARIANT = 'yolo11x.pt'

    # 3. 训练超参数
    EPOCHS = 100
    IMAGE_SIZE = 640
    BATCH_SIZE = 8  # 根据您的GPU显存进行调整
    DEVICE = '0'    # 使用第一个GPU进行训练。如果没有GPU，请设置为 'cpu'

    # --- 开始训练 ---
    train_yolov11_weak_aug(
        data_yaml_path=DATA_YAML_PATH,
        model_variant=MODEL_VARIANT,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )