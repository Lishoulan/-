import os
from ultralytics import YOLO
from glob import glob

# --- 1. 配置您的模型和图片路径 ---

# 模型路径：指向您训练好的 .pt 权重文件
MODEL_PATH = r'F:\ultralytics-main\runs\detect\train6augdata\weights\best.pt' 


# 图片路径：指向包含新图片的文件夹
# 代码会自动处理该文件夹下所有的 .jpg, .png, .jpeg 图片
IMAGE_DIR = r'F:\ultralytics-main\dataset\test'

# 结果保存路径：推理结果（带标注框的图片）将保存在这里
OUTPUT_DIR = r'F:\ultralytics-main\dataset\testput'


# --- 2. 加载您训练好的 YOLOv11 模型 ---

# 加载模型，如果您的模型是基于 ultralytics 训练的，这行代码通常无需修改
try:
    model = YOLO(MODEL_PATH)
    print(f"成功加载模型： {MODEL_PATH}")
except Exception as e:
    print(f"错误：无法加载模型，请检查路径和模型文件。")
    print(f"详细错误: {e}")
    exit()

# --- 3. 执行推理 ---

# 在指定的图片文件夹上运行推理
# confidence (conf): 置信度阈值，只显示高于此值的检测结果
# save=True: 将带标注框的图片保存到磁盘
# project 和 name: 定义了结果的保存目录
try:
    print(f"开始在文件夹 '{IMAGE_DIR}' 上进行推理...")
    results = model.predict(source=IMAGE_DIR, conf=0.5, save=True, project=OUTPUT_DIR)
    print("推理完成！")
    
    # results 对象包含了详细的检测信息，您可以按需处理
    # 例如，打印每张图片检测到的目标信息
    for r in results:
        print("-" * 30)
        print(f"图片路径: {r.path}")
        print(f"检测到的目标数量: {len(r.boxes)}")
        # r.show()  # 如果在桌面环境中，可以取消注释以显示结果图片
        # r.save(filename='result.jpg') # 或者单独保存处理后的图片

    # 获取结果保存的具体路径
    # ultralytics 会自动创建一个带递增编号的子文件夹
    # result_save_dir = results[0].save_dir # 获取实际保存目录
    print(f"结果已保存到 '{OUTPUT_DIR}' 目录下。")

except Exception as e:
    print(f"推理过程中发生错误。")
    print(f"详细错误: {e}")