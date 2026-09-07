import os
import shutil
import yaml
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_yolo_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_state=42):
    """
    将images和labels文件夹转换为YOLO格式数据集
    
    Args:
        images_dir: 原始图片文件夹路径
        labels_dir: 原始标签文件夹路径（YOLO格式的txt文件）
        output_dir: 输出数据集根目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_state: 随机种子
    """
    
    # 创建输出目录结构
    output_path = Path(output_dir)
    train_images_dir = output_path / 'images' / 'train'
    train_labels_dir = output_path / 'labels' / 'train'
    val_images_dir = output_path / 'images' / 'val'
    val_labels_dir = output_path / 'labels' / 'val'
    test_images_dir = output_path / 'images' / 'test'
    test_labels_dir = output_path / 'labels' / 'test'
    
    # 创建所有必要的文件夹
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, 
                     val_labels_dir, test_images_dir, test_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片文件
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 获取所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_path.glob(f'*{ext}')))
        image_files.extend(list(images_path.glob(f'*{ext.upper()}')))
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 检查标签文件是否存在
    valid_files = []
    missing_labels = []
    
    for img_path in image_files:
        label_file = labels_path / f"{img_path.stem}.txt"
        if label_file.exists():
            # 检查标签文件是否为空或只有背景（无目标）
            if label_file.stat().st_size > 0:
                # 检查是否至少有一行有效标签
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    if any(line.strip() for line in lines):
                        valid_files.append(img_path)
                    else:
                        missing_labels.append(img_path.stem)
            else:
                missing_labels.append(img_path.stem)
        else:
            missing_labels.append(img_path.stem)
    
    if missing_labels:
        print(f"警告: {len(missing_labels)} 个图片没有有效的标签文件，将被跳过")
        print(f"示例: {missing_labels[:5]}")
    
    print(f"有效图片数量: {len(valid_files)}")
    
    if len(valid_files) == 0:
        raise ValueError("没有找到有效的图片-标签配对")
    
    # 按照比例划分数据集
    train_files, temp_files = train_test_split(
        valid_files, 
        train_size=train_ratio,
        random_state=random_state
    )
    
    # 计算验证集和测试集的比例
    if val_ratio + test_ratio > 0:
        val_ratio_adj = val_ratio / (val_ratio + test_ratio)
        val_files, test_files = train_test_split(
            temp_files,
            train_size=val_ratio_adj,
            random_state=random_state
        )
    else:
        val_files = []
        test_files = []
    
    print(f"数据集划分: 训练集 {len(train_files)}, 验证集 {len(val_files)}, 测试集 {len(test_files)}")
    
    # 复制文件到对应目录
    def copy_files(file_list, target_images_dir, target_labels_dir):
        for img_path in file_list:
            # 复制图片
            shutil.copy2(img_path, target_images_dir / img_path.name)
            
            # 复制对应的标签文件
            label_file = labels_path / f"{img_path.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, target_labels_dir / label_file.name)
    
    print("正在复制训练集文件...")
    copy_files(train_files, train_images_dir, train_labels_dir)
    
    if val_files:
        print("正在复制验证集文件...")
        copy_files(val_files, val_images_dir, val_labels_dir)
    
    if test_files:
        print("正在复制测试集文件...")
        copy_files(test_files, test_images_dir, test_labels_dir)
    
    # 生成data.yaml文件
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,  # 1 class (leaf)
        'names': ['leaf']
    }
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
    
    print(f"数据集创建完成！data.yaml已保存到: {yaml_path}")
    print("\n数据集结构:")
    print(output_path)
    print(f"├── images/")
    print(f"│   ├── train/ ({len(train_files)} files)")
    if val_files:
        print(f"│   ├── val/ ({len(val_files)} files)")
    if test_files:
        print(f"│   └── test/ ({len(test_files)} files)")
    print(f"├── labels/")
    print(f"│   ├── train/ ({len(train_files)} files)")
    if val_files:
        print(f"│   ├── val/ ({len(val_files)} files)")
    if test_files:
        print(f"│   └── test/ ({len(test_files)} files)")
    print(f"└── data.yaml")
    
    return output_path

def verify_dataset(data_dir):
    """
    验证数据集完整性
    """
    data_dir = Path(data_dir)
    print("\n验证数据集...")
    
    for split in ['train', 'val', 'test']:
        images_dir = data_dir / 'images' / split
        labels_dir = data_dir / 'labels' / split
        
        if not images_dir.exists():
            continue
            
        image_files = list(images_dir.glob('*'))
        label_files = list(labels_dir.glob('*.txt'))
        
        print(f"{split}集: {len(image_files)} 张图片, {len(label_files)} 个标签文件")
        
        # 检查配对
        image_names = {f.stem for f in image_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}}
        label_names = {f.stem for f in label_files}
        
        missing_labels = image_names - label_names
        if missing_labels:
            print(f"警告: {split}集缺少 {len(missing_labels)} 个标签文件")

# 使用示例
if __name__ == "__main__":
    # 设置路径
    images_folder = "images"  # 原始图片文件夹
    labels_folder = "labels"  # 原始标签文件夹
    output_folder = "uav_dataset"  # 输出数据集文件夹
    
    # 创建YOLO格式数据集
    try:
        dataset_path = create_yolo_dataset(
            images_dir=images_folder,
            labels_dir=labels_folder,
            output_dir=output_folder,
            train_ratio=0.7,  # 训练集70%
            val_ratio=0.2,    # 验证集20%
            test_ratio=0.1,   # 测试集10%
            random_state=42   # 随机种子，保证可重复性
        )
        
        # 验证数据集
        verify_dataset(dataset_path)
        
        print("\n成功！数据集已准备好进行YOLO训练。")
        print(f"data.yaml文件路径: {dataset_path / 'data.yaml'}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")