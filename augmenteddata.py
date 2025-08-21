import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
import shutil

# ==============================================================================
# 1. 配置区 - 请根据您的项目修改这里
# ==============================================================================

# 原始数据路径
IMAGE_DIR = 'dataset/images'  # 请确保这里是你的实际路径
LABEL_DIR = 'dataset/labels'  # 请确保这里是你的实际路径

# 增强后数据的保存路径 (建议新建文件夹，避免覆盖原始数据)
AUG_IMAGE_DIR = 'dataset/augmented/images'
AUG_LABEL_DIR = 'dataset/augmented/labels'

CLASS_NAMES = [
    'hanyan', 'zhadian', 'handon', 'hangao', 'hanzha', 
    'duanhan', 'hanchuan', 'hanpian'
]

# 定义哪些是“少数类”，需要被大力度增强
MINORITY_CLASSES = [
    'zhadian', 'hangao', 'hanzha', 'duanhan', 'hanchuan', 'hanpian'
]

# 对每一张包含“少数类”的图片，我们为它生成多少张增强后的图片
AUGMENTATIONS_PER_MINORITY_IMAGE = 100

# ==============================================================================
# 2. 修正后的数据增强流水线
# ==============================================================================

# 获取少数类的 class_id
minority_class_ids = [i for i, name in enumerate(CLASS_NAMES) if name in MINORITY_CLASSES]

# --- 为“少数类”定制的强力增强流水线 (已移除 CoarseDropout 和 ShiftScaleRotate) ---
strong_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
    A.GaussNoise(p=0.5), 
    A.RandomGamma(p=0.5)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1))

# --- 为“多数类”定制的轻微增强流水线 ---
light_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# ==============================================================================
# 3. 辅助函数 (包含中文路径读写修复)
# ==============================================================================

def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """正确读取包含非ASCII字符路径的图像"""
    try:
        raw_data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(raw_data, flags)
        return img
    except Exception as e:
        print(f"读取文件时出错 {path}: {e}")
        return None

def imwrite_unicode(path, img):
    """正确写入包含非ASCII字符路径的图像"""
    try:
        _, buf = cv2.imencode('.jpg', img)
        buf.tofile(path)
        return True
    except Exception as e:
        print(f"写入文件时出错 {path}: {e}")
        return False

def read_yolo_labels(label_path):
    """读取YOLO格式的标注文件"""
    bboxes = []
    class_labels = []
    if not os.path.exists(label_path):
        return bboxes, class_labels
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            bboxes.append(coords)
            class_labels.append(class_id)
    return bboxes, class_labels

def save_yolo_labels(label_path, bboxes, class_labels):
    """保存为YOLO格式的标注文件"""
    with open(label_path, 'w', encoding='utf-8') as f:
        for bbox, class_id in zip(bboxes, class_labels):
            f.write(f"{int(class_id)} {' '.join(map(str, bbox))}\n")


# ==============================================================================
# 4. 主执行脚本
# ==============================================================================

def main():
    print("开始进行针对性数据增强...")
    
    os.makedirs(AUG_IMAGE_DIR, exist_ok=True)
    os.makedirs(AUG_LABEL_DIR, exist_ok=True)

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_name in tqdm(image_files, desc="处理图像"):
        image_path = os.path.join(IMAGE_DIR, image_name)
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(LABEL_DIR, label_name)

        image = imread_unicode(image_path)
        
        if image is None:
            print(f"警告：无法读取图像 {image_path}，已跳过。")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes, class_labels = read_yolo_labels(label_path)
        
        if not bboxes:
            continue

        is_minority_image = any(class_id in minority_class_ids for class_id in class_labels)
        
        transform = strong_aug if is_minority_image else light_aug
        num_augmentations = AUGMENTATIONS_PER_MINORITY_IMAGE if is_minority_image else 1
        aug_type_prefix = "strong" if is_minority_image else "light"

        for i in range(num_augmentations):
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                if augmented['bboxes']:
                    aug_image_name = f"{os.path.splitext(image_name)[0]}_aug_{aug_type_prefix}_{i}.jpg"
                    aug_label_name = f"{os.path.splitext(label_name)[0]}_aug_{aug_type_prefix}_{i}.txt"
                    
                    aug_image_path = os.path.join(AUG_IMAGE_DIR, aug_image_name)
                    aug_label_path = os.path.join(AUG_LABEL_DIR, aug_label_name)
                    
                    imwrite_unicode(aug_image_path, cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR))
                    save_yolo_labels(aug_label_path, augmented['bboxes'], augmented['class_labels'])
            except Exception as e:
                print(f"处理 {image_name} (增强) 时发生错误: {e}")
        
        shutil.copy(image_path, os.path.join(AUG_IMAGE_DIR, image_name))
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(AUG_LABEL_DIR, label_name))

    print("-" * 30)
    print("数据增强完成!")
    print(f"增强后的图像保存在: {AUG_IMAGE_DIR}")
    print(f"增强后的标签保存在: {AUG_LABEL_DIR}")
    print("请使用这两个新文件夹中的数据来重新训练您的模型。")

if __name__ == '__main__':
    main()