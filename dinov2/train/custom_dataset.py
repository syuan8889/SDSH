import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

class CustomImageDataset(Dataset):
    """
    自定义图像数据集类
    
    数据集组织格式：
    root/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── ...
    
    Args:
        root (str): 数据集根目录路径
        train (bool): True表示训练模式，False表示测试模式
        train_ratio (float): 训练数据比例，范围[0, 1]，默认0.8
        transform (callable, optional): 图像变换函数
        seed (int): 随机种子，确保训练/测试分割的一致性
    """
    
    def __init__(
        self, 
        root: str, 
        train: bool = True, 
        train_ratio: float = 0.8,
        transform: Optional[callable] = None,
        seed: int = 42
    ):
        self.root = root
        self.train = train
        self.train_ratio = train_ratio
        self.transform = transform
        
        # 设置随机种子确保可重复性
        random.seed(seed)
        
        # 获取类别列表
        self.classes = self._get_classes()
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # 构建数据列表
        self.data = self._build_data_list()
        
        logger.info(f"Dataset initialized: {len(self.data)} samples, "
                   f"{len(self.classes)} classes, "
                   f"{'train' if train else 'test'} mode")
    
    def _get_classes(self) -> List[str]:
        """获取所有类别名称"""
        classes = []
        for item in os.listdir(self.root):
            item_path = os.path.join(self.root, item)
            if os.path.isdir(item_path):
                classes.append(item)
        return sorted(classes)
    
    def _build_data_list(self) -> List[Tuple[str, int]]:
        """构建数据列表，包含图像路径和类别索引"""
        data = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.root, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # 获取该类别下的所有图像文件
            image_files = []
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                if self._is_image_file(file_path):
                    image_files.append(file_path)
            
            # 按文件名排序确保一致性
            image_files.sort()
            
            # 计算训练/测试分割点
            total_images = len(image_files)
            train_count = int(total_images * self.train_ratio)
            
            if self.train:
                # 训练模式：取前train_ratio比例的数据
                selected_files = image_files[:train_count]
            else:
                # 测试模式：取剩余的数据
                selected_files = image_files[train_count:]
            
            # 添加到数据列表
            for file_path in selected_files:
                data.append((file_path, class_idx))
        
        return data
    
    def _is_image_file(self, file_path: str) -> bool:
        """判断文件是否为图像文件"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
        _, ext = os.path.splitext(file_path.lower())
        return ext in image_extensions
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取单个样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple: (image, class_idx) 图像张量和类别索引
        """
        image_path, class_idx = self.data[idx]
        
        # 读取图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # 返回一个黑色图像作为fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # 应用变换
        if self.transform is not None:
            image = self.transform(image)
        
        return image, class_idx, idx
    
    def get_class_names(self) -> List[str]:
        """获取类别名称列表"""
        return self.classes
    
    def get_class_to_idx(self) -> dict:
        """获取类别名称到索引的映射"""
        return self.class_to_idx.copy()
    
    def get_class_distribution(self) -> dict:
        """获取各类别的样本数量分布"""
        distribution = {}
        for _, class_idx in self.data:
            class_name = self.classes[class_idx]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution


# 使用示例
if __name__ == "__main__":
    # 示例用法
    from torchvision import transforms
    
    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建训练数据集
    train_dataset = CustomImageDataset(
        root="/home/ouc/data1/qiaoshishi/datasets/AID",
        train=True,
        train_ratio=0.5,
        transform=transform
    )
    
    # 创建测试数据集
    test_dataset = CustomImageDataset(
        root="/home/ouc/data1/qiaoshishi/datasets/AID",
        train=False,
        train_ratio=0.5,
        transform=transform
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"类别数量: {len(train_dataset.classes)}")
    print(f"类别分布: {train_dataset.get_class_distribution()}")
    
    # 测试获取样本
    image, class_idx = train_dataset[0]
    print(f"图像形状: {image.shape}")
    print(f"类别索引: {class_idx}")
    print(f"类别名称: {train_dataset.classes[class_idx]}")
