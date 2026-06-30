import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms 
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# ==========================================
# 1. 单类别ViT数据集 - 适配8类别版本
# ==========================================

class SingleClassViTDataset(torch.utils.data.Dataset):
    """单类别图片ViT数据集 - 从train文件夹按比例划分训练/测试（适配8类别）"""
    def __init__(self, root, split='train', img_transform=None, train_ratio=0.8, shuffle=True, seed=42):
        """
        Args:
            root: 数据根目录（包含A、B、C、D、E、F、G、H等类别文件夹）
            split: 'train' 或 'test' - 指定返回训练集还是测试集
            img_transform: 图像变换
            train_ratio: 训练集划分比例（默认0.8）
            shuffle: 是否打乱数据
            seed: 随机种子，确保每次划分一致
        """
        self.root = root
        self.split = split
        self.img_transform = img_transform
        self.train_ratio = train_ratio
        self.seed = seed
        
        # 设置随机种子，确保划分一致性
        random.seed(seed)
        
        # 自动获取类别（从文件夹名，适配A-H 8类）
        self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # 校验类别数量（提示8类）
        if len(self.classes) != 8:
            print(f"[WARN] 检测到 {len(self.classes)} 个类别，预期8类（A-H）！")
            print(f"[WARN] 当前识别的类别：{self.classes}")
        
        # 初始化存储
        self.samples = []
        self.labels = []
        
        # 按类别处理，确保每个类别都按比例划分
        for class_name in self.classes:
            class_folder = os.path.join(root, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # 获取该类别所有图片（支持更多格式）
            image_files = []
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp']:
                image_files.extend([f for f in os.listdir(class_folder) if f.lower().endswith(ext)])
            
            if not image_files:
                print(f"[WARN] No images found in class folder: {class_folder}")
                continue
            
            # 按数字顺序排序（如果有数字命名）
            try:
                # 尝试按数字顺序排序
                image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0])) or '0'))
            except:
                # 如果文件名不是纯数字，则按字母顺序排序
                image_files.sort()
            
            # 计算划分点
            total_images = len(image_files)
            train_size = int(total_images * train_ratio)
            
            if split == 'train':
                # 训练集：取前 train_size 张图片
                selected_files = image_files[:train_size]
            else:  # split == 'test'
                # 测试集：取剩余的图片
                selected_files = image_files[train_size:]
            
            print(f"[DEBUG] Class {class_name}: {total_images} total, {len(selected_files)} for {split}")
            
            # 如果需要打乱且是训练集
            if shuffle and split == 'train':
                random.shuffle(selected_files)
            
            # 添加到样本列表
            for img_file in selected_files:
                img_path = os.path.join(class_folder, img_file)
                self.samples.append({
                    'image': img_path,
                    'label': class_idx,
                    'class_name': class_name
                })
                self.labels.append(class_idx)
        
        print(f"[INFO] {split} set: {len(self.samples)} samples, {len(self.classes)} classes")
        print(f"[INFO] Classes: {self.classes}")
        
        # 显示每个类别的样本分布
        self._show_class_distribution()
    
    def _show_class_distribution(self):
        """显示每个类别的样本数（重点展示8类分布）"""
        class_counts = {}
        for sample in self.samples:
            class_name = sample['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"[INFO] {self.split} set class distribution (8类预期):")
        # 按A-H顺序打印，确保直观
        expected_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        for cls in expected_classes:
            if cls in class_counts:
                print(f"  {cls}: {class_counts[cls]} samples")
            else:
                print(f"  {cls}: 0 samples (缺失！)")
        # 打印额外的类别（如果有）
        extra_classes = [cls for cls in class_counts if cls not in expected_classes]
        if extra_classes:
            for cls in extra_classes:
                print(f"  {cls}: {class_counts[cls]} samples (非预期类别！)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # 加载图像
        img_path = sample_info['image']
        img = Image.open(img_path).convert('RGB')
        
        if self.img_transform:
            img = self.img_transform(img)
        
        label = sample_info['label']
        
        return img, label

# ==========================================
# 2. ViT模型包装器（无需修改）
# ==========================================

class ViTWrapper(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT
        self.vit = vit_b_16(weights=weights).to(device)
        # 移除预训练的分类头
        self.vit.heads = nn.Identity()

    def forward(self, img: torch.Tensor):
        return self.vit(img)

def build_visual_encoder(device: torch.device):
    vit_wrapper = ViTWrapper(device).to(device)
    return vit_wrapper, 768

# ==========================================
# 3. 分类头（自动适配8类别，无需修改）
# ==========================================

class SimpleClassifier(nn.Module):
    """简单的分类器，基于ViT特征（自动适配类别数）"""
    def __init__(self, vit_dim=768, num_classes=8):  # 默认8类
        super().__init__()
        self.num_classes = num_classes
        
        # 分类头（适配8类，调整了隐藏层维度）
        self.classifier = nn.Sequential(
            nn.Linear(vit_dim, 1024),  # 从512→1024，提升8类分类能力
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),  # 增大dropout，防止8类过拟合
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, features):
        # features: (B, 768)
        logits = self.classifier(features)
        return logits

# ==========================================
# 4. 评估函数（无需修改）
# ==========================================

@torch.no_grad()
def evaluate(vit_model, classifier, loader, device, criterion, split_name: str = "Eval"):
    """评估函数"""
    vit_model.eval()
    classifier.eval()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"[{split_name}]", ncols=100)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 提取特征并分类
        features = vit_model(images)  # (B, 768)
        logits = classifier(features)
        loss = criterion(logits, labels)

        pred = logits.argmax(dim=1)
        total_loss += loss.item()
        total_samples += labels.size(0)
        correct += (pred == labels).sum().item()

        acc_batch = (pred == labels).float().mean().item() * 100.0
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc_batch:.2f}%")

    avg_loss = total_loss / len(loader)
    avg_acc = (correct / total_samples) * 100.0
    return avg_loss, avg_acc

# ==========================================
# 5. 主函数 - 适配8类别版本
# ==========================================

def main():
    # 设置全局随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    parser = argparse.ArgumentParser(description='ViT training for 8 classes (A-H) with 80/20 split')
    parser.add_argument('--epochs', type=int, default=20,  # 增加训练轮数（适配8类）
                        help='Number of training epochs (20 for 8 classes)')
    parser.add_argument('--lr', type=float, default=1e-4,  # 降低学习率（8类更稳定）
                        help='Base learning rate (1e-4 for 8 classes)')
    parser.add_argument('--min_lr', type=float, default=1e-6, 
                        help='Minimum learning rate for scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=3,  # 增加warmup轮数
                        help='Epochs for linear warmup (3 for 8 classes)')
    
    parser.add_argument('--batch_size', type=int, default=16,  # 降低批次大小（适配8类+显存）
                        help='Batch size (16 recommended for 8 classes)')
    parser.add_argument('--data_root', type=str, default='data',
                        help='数据根目录，包含train文件夹（A-H 8类）')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据集划分参数
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集划分比例（默认0.8）')
    parser.add_argument('--val_split', action='store_true', default=True,  # 默认启用验证集
                        help='是否从训练集中划分验证集（8类建议启用）')
    parser.add_argument('--val_ratio', type=float, default=0.15,  # 验证集比例提升到15%
                        help='验证集划分比例（8类建议0.15）')
    
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"==================================================================")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Training for 8 CLASSES (A-H)")
    print(f"[INFO] Training ratio: {args.train_ratio * 100}%")
    print(f"[INFO] Testing ratio: {(1 - args.train_ratio) * 100}%")
    print(f"[INFO] Validation split: {args.val_split} (ratio: {args.val_ratio})")
    print(f"==================================================================")

    # --- 1. 数据变换（适配8类，增强数据增强）---
    img_transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),  # 新增垂直翻转（8类需要更多增强）
        transforms.RandomRotation(15),  # 增大旋转角度
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),  # 增强色彩变换
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),  # 新增随机裁剪
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),  # 新增随机擦除（防止过拟合）
    ])
    
    img_transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 2. 创建数据集 ---
    train_root = os.path.join(args.data_root, 'train')
    
    # 检查train文件夹是否存在
    if not os.path.exists(train_root):
        print(f"[ERROR] Train folder not found: {train_root}")
        print("[INFO] Expected directory structure for 8 classes (A-H):")
        print("  data/")
        print("  └── train/")
        print("      ├── A/")
        print("      ├── B/")
        print("      ├── C/")
        print("      ├── D/")
        print("      ├── E/")
        print("      ├── F/")
        print("      ├── G/")
        print("      └── H/")
        return
    
    print("\n" + "="*70)
    print("[INFO] Creating training set (80% of train folder) for 8 classes...")
    print("="*70)
    
    # 创建训练集
    train_dataset = SingleClassViTDataset(
        root=train_root, 
        split='train',
        img_transform=img_transform_train, 
        train_ratio=args.train_ratio,
        shuffle=True,
        seed=seed
    )
    
    print("\n" + "="*70)
    print("[INFO] Creating test set (20% of train folder) for 8 classes...")
    print("="*70)
    
    # 创建测试集
    test_dataset = SingleClassViTDataset(
        root=train_root, 
        split='test',
        img_transform=img_transform_test, 
        train_ratio=args.train_ratio,
        shuffle=False,
        seed=seed
    )
    
    # 创建验证集（8类建议启用）
    if args.val_split:
        print(f"\n[INFO] Creating validation set from training data (ratio: {args.val_ratio})...")
        val_size = int(len(train_dataset) * args.val_ratio)
        train_size = len(train_dataset) - val_size
        
        # 固定种子确保划分一致性
        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size], generator=generator
        )
        
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, 
                                 shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        
        print(f"[INFO] Training subset: {len(train_subset)} samples")
        print(f"[INFO] Validation set: {len(val_subset)} samples")
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                 shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        val_loader = None

    # 测试集DataLoader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    # 汇总信息（重点展示8类）
    print("\n" + "="*70)
    print("[INFO] Dataset Summary (8 Classes Expected)")
    print("="*70)
    print(f"Total classes detected: {len(train_dataset.classes)} (expected 8)")
    print(f"Classes detected: {train_dataset.classes}")
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    
    if args.val_split and val_loader:
        print(f"Validation set: {len(val_subset)} samples")
    
    # 计算并显示8类的样本分布
    total_samples_all_classes = 0
    expected_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    print("\n[INFO] Sample count per class (8 classes):")
    for cls in expected_classes:
        class_folder = os.path.join(train_root, cls)
        if os.path.exists(class_folder):
            image_files = []
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp']:
                image_files.extend([f for f in os.listdir(class_folder) if f.lower().endswith(ext)])
            total_samples_all_classes += len(image_files)
            print(f"  {cls}: {len(image_files)} images")
        else:
            print(f"  {cls}: 0 images (folder missing!)")
    
    # 打印额外类别（如果有）
    all_folders = [d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))]
    extra_classes = [cls for cls in all_folders if cls not in expected_classes]
    if extra_classes:
        print("\n[WARN] Extra classes detected (not A-H):")
        for cls in extra_classes:
            class_folder = os.path.join(train_root, cls)
            image_files = [f for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f))]
            print(f"  {cls}: {len(image_files)} images")
    
    print(f"\nTotal images in 8 target classes: {total_samples_all_classes}")
    print(f"Train images ({args.train_ratio*100}%): {int(total_samples_all_classes * args.train_ratio)} expected")
    print(f"Test images ({100 - args.train_ratio*100}%): {int(total_samples_all_classes * (1 - args.train_ratio))} expected")
    print("="*70)

    # --- 3. 模型构建（自动适配8类）---
    print("\n[INFO] Building ViT model for 8 classes...")
    vit_model, vit_dim = build_visual_encoder(device)
    
    # 分类头自动适配检测到的类别数（确保是8类）
    num_classes = len(train_dataset.classes)
    classifier = SimpleClassifier(
        vit_dim=vit_dim,
        num_classes=num_classes
    ).to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in vit_model.parameters())
    trainable_params = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
    print(f"[INFO] ViT Model - Total params: {total_params:,} | Trainable: {trainable_params:,}")
    
    classifier_params = sum(p.numel() for p in classifier.parameters())
    print(f"[INFO] Classifier (8 classes) - Trainable params: {classifier_params:,}")

    # --- 4. 优化器（适配8类）---
    ft_vit = True
    
    params_group_high_lr = list(classifier.parameters())
    params_group_low_lr = []
    
    if ft_vit:
        print("[INFO] Adding ViT parameters with 0.1x learning rate (8 classes).")
        params_group_low_lr += list(vit_model.parameters())

    optimizer = optim.AdamW([
        {'params': params_group_high_lr, 'lr': args.lr},
        {'params': params_group_low_lr, 'lr': args.lr * 0.1}  # 降低ViT的学习率（8类更稳定）
    ], weight_decay=0.05)  # 降低权重衰减（8类）

    # --- 5. 学习率调度器（适配8类）---
    scheduler_warmup = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_epochs
    )
    scheduler_cosine = CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[scheduler_warmup, scheduler_cosine], 
        milestones=[args.warmup_epochs]
    )

    criterion = nn.CrossEntropyLoss()

    # 记录最佳表现
    best_acc = 0.0
    best_epoch = -1
    best_iter = -1
    save_best_path = 'checkpoints/vit_best_8classes.pth'  # 8类模型保存路径

    # --- 6. 训练循环（适配8类）---
    print("\n" + "="*70)
    print("[INFO] Starting training for 8 classes (A-H)...")
    print("="*70)
    
    for epoch in range(args.epochs):
        vit_model.train() if ft_vit else vit_model.eval()
        classifier.train()

        total_loss = 0
        correct = 0
        total_samples = 0
        
        current_lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] LR={current_lr:.2e}", ncols=120)

        for it, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward
            features = vit_model(images)
            logits = classifier(features)
            loss = criterion(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算准确率
            pred = logits.argmax(dim=1)
            acc_batch = (pred == labels).float().mean().item() * 100.0

            total_loss += loss.item()
            total_samples += labels.size(0)
            correct += (pred == labels).sum().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc_batch:.2f}%")

            # 每50个iteration评估一次（8类更频繁评估）
            if (it + 1) % 50 == 0:
                if args.val_split and val_loader is not None:
                    eval_loader = val_loader
                    split_name = f"Val(it={it+1})"
                else:
                    eval_loader = test_loader
                    split_name = f"Test(it={it+1})"
                
                eval_loss, eval_acc = evaluate(
                    vit_model, classifier, eval_loader, device, criterion,
                    split_name=split_name
                )
                print(
                    f"[Epoch {epoch+1} Iter {it+1}] {split_name} Loss: {eval_loss:.4f}, "
                    f"Acc: {eval_acc:.2f}%"
                )

                # 保存最佳模型
                if eval_acc > best_acc:
                    best_acc = eval_acc
                    best_epoch = epoch + 1
                    best_iter = it + 1
                    save_dir = os.path.dirname(save_best_path)
                    if save_dir != '':
                        os.makedirs(save_dir, exist_ok=True)
                    torch.save({
                        'epoch': best_epoch,
                        'iter': best_iter,
                        'eval_acc': best_acc,
                        'eval_loss': eval_loss,
                        'classifier': classifier.state_dict(),
                        'vit_model': vit_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'args': vars(args),
                        'classes': train_dataset.classes,
                        'class_to_idx': train_dataset.class_to_idx,  # 保存类别映射
                    }, save_best_path)
                    print(f"[BEST] New best Acc: {best_acc:.2f}% at epoch {best_epoch}, iter {best_iter}")

                # 评估后重新切回训练模式
                vit_model.train() if ft_vit else vit_model.eval()
                classifier.train()

        # 更新学习率
        scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = (correct / total_samples) * 100.0
        print(f"==> Epoch {epoch+1} Finished. Avg Loss: {avg_train_loss:.4f}, Avg Acc: {avg_train_acc:.2f}%")

    # 训练结束后在测试集上做一次最终评估
    print("\n" + "="*70)
    print("[INFO] Final evaluation on test set (20% of data) for 8 classes...")
    print("="*70)
    
    test_loss, test_acc = evaluate(vit_model, classifier, test_loader, device, criterion, split_name="Test-Final")
    print(f"\n[FINAL] Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

    # 最后一轮评估也参与best选择
    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = args.epochs
        best_iter = -1
        save_dir = os.path.dirname(save_best_path)
        if save_dir != '':
            os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'epoch': best_epoch,
            'iter': best_iter,
            'eval_acc': best_acc,
            'eval_loss': test_loss,
            'classifier': classifier.state_dict(),
            'vit_model': vit_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
            'classes': train_dataset.classes,
            'class_to_idx': train_dataset.class_to_idx,
        }, save_best_path)
        print(f"[BEST-FINAL] New best Acc: {best_acc:.2f}% at epoch {best_epoch}")

    print(f"\n[SUMMARY] Best evaluation accuracy (8 classes): {best_acc:.2f}% at epoch {best_epoch}")
    print(f"[SUMMARY] Trained on {len(train_dataset)} samples ({args.train_ratio*100}% of data)")
    print(f"[SUMMARY] Tested on {len(test_dataset)} samples ({(1-args.train_ratio)*100}% of data)")
    print(f"[SUMMARY] Classes trained: {train_dataset.classes}")
    print("[DONE] Training finished for 8 classes (A-H).")

if __name__ == '__main__':
    main()