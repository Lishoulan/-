from ultralytics import YOLO

def validate_yolov11_model(weights_path, data_yaml_path, imgsz=640, batch_size=16, device='cpu'):
    """
    在指定的数据集分割上验证YOLOv11分割模型的性能。

    参数:
    - weights_path (str): 指向您训练好的模型权重文件 (.pt) 的路径。
    - data_yaml_path (str): 指向您的数据集 .yaml 文件的路径。
    - imgsz (int): 验证时输入的图像尺寸。
    - batch_size (int): 验证时的批处理大小。
    - device (str): 验证设备, 'cpu' 或 '0' (对于第一个GPU)。
    """
    try:
        # 加载您训练好的模型
        model = YOLO(weights_path)

        # 运行验证
        # 函数会返回一个包含所有性能指标的字典
        metrics = model.val(
            data=data_yaml_path,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            plots=True  # 保存额外的可视化图表，如混淆矩阵
        )

        # 打印关键的性能指标
        print("\n--- 验证结果 ---")
        print(f"目标检测任务 mAP50-95: {metrics.box.map:.4f}")
        print(f"目标检测任务 mAP50: {metrics.box.map50:.4f}")
        print("\n详细指标已保存。")
        # 结果和图表将保存在 'runs/segment/val' 目录下

    except Exception as e:
        print(f"验证过程中发生错误: {e}")

if __name__ == '__main__':
    # --- 请根据您的实际情况修改以下参数 ---

    # 1. 训练好的模型权重路径
    #    这通常是训练完成后 'runs/segment/trainX/weights/' 目录下的 'best.pt' 文件。
    WEIGHTS_PATH = r'F:\ultralytics-main\runs\detect\train6\weights\best.pt'  # <-- 修改这里

    # 2. 数据集YAML文件的路径
    DATA_YAML_PATH = 'dianchi.yaml'  # <-- 修改这里



    # 3. 验证参数
    IMAGE_SIZE = 640
    DEVICE = '0'  # 使用GPU进行验证，若无则设为 'cpu'


    # --- 开始验证 ---
    validate_yolov11_model(
        weights_path=WEIGHTS_PATH,
        data_yaml_path=DATA_YAML_PATH,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
    )