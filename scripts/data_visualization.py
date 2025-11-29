
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
VisDrone2019数据集可视化脚本
功能：分析和可视化数据集分布情况
'''

import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# VisDrone2019类别名称
CLASS_NAMES = [
    'pedestrian',      # 0
    'people',          # 1
    'bicycle',         # 2
    'car',             # 3
    'van',             # 4
    'truck',           # 5
    'tricycle',        # 6
    'awning-tricycle', # 7
    'bus',             # 8
    'motor'            # 9
]

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='VisDrone2019数据集可视化')
    parser.add_argument('--data_dir', type=str, default='./datasets/visdrone2019', 
                      help='数据集根目录路径')
    parser.add_argument('--split', type=str, default='train', 
                      choices=['train', 'val', 'test'],
                      help='数据集划分')
    parser.add_argument('--output_dir', type=str, default='./visualization', 
                      help='可视化结果输出目录')
    return parser.parse_args()

def count_classes(label_dir):
    """
    统计每个类别的样本数量
    
    参数：
        label_dir: 标注文件目录路径
    
    返回：
        class_counts: 类别计数字典
        total_objects: 总目标数量
    """
    class_counts = {i: 0 for i in range(len(CLASS_NAMES))}
    total_objects = 0
    
    # 遍历所有标注文件
    label_files = list(label_dir.glob('*.txt'))
    pbar = tqdm(label_files, desc=f'Counting classes in {label_dir.name}')
    
    for label_file in pbar:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    # YOLO格式：class x_center y_center width height
                    cls = int(line.split()[0])
                    if 0 <= cls < len(CLASS_NAMES):
                        class_counts[cls] += 1
                        total_objects += 1
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
            continue
    
    return class_counts, total_objects

def analyze_box_sizes(label_dir):
    """
    分析目标边界框尺寸分布
    
    参数：
        label_dir: 标注文件目录路径
    
    返回：
        box_sizes: 边界框尺寸列表 [(width, height), ...]
        class_sizes: 按类别划分的边界框尺寸字典
    """
    box_sizes = []
    class_sizes = {i: [] for i in range(len(CLASS_NAMES))}
    
    # 遍历所有标注文件
    label_files = list(label_dir.glob('*.txt'))
    pbar = tqdm(label_files, desc=f'Analyzing box sizes in {label_dir.name}')
    
    for label_file in pbar:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    # YOLO格式：class x_center y_center width height
                    parts = line.split()
                    cls = int(parts[0])
                    if 0 <= cls < len(CLASS_NAMES):
                        width = float(parts[3])
                        height = float(parts[4])
                        box_sizes.append((width, height))
                        class_sizes[cls].append((width, height))
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
            continue
    
    return box_sizes, class_sizes

def plot_class_distribution(class_counts, total_objects, output_path):
    """
    绘制类别分布柱状图
    
    参数：
        class_counts: 类别计数字典
        total_objects: 总目标数量
        output_path: 输出图像路径
    """
    plt.figure(figsize=(12, 6))
    
    # 准备数据
    classes = list(range(len(CLASS_NAMES)))
    counts = [class_counts[cls] for cls in classes]
    
    # 绘制柱状图
    bars = plt.bar(classes, counts, color='skyblue')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 50, 
                f'{height}', ha='center', va='bottom', fontsize=10)
    
    # 设置图表属性
    plt.title(f'Class Distribution (Total: {total_objects} objects)', fontsize=14)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Objects', fontsize=12)
    plt.xticks(classes, CLASS_NAMES, rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"类别分布图表已保存到: {output_path}")

def plot_box_size_distribution(box_sizes, class_sizes, output_dir):
    """
    绘制边界框尺寸分布
    
    参数：
        box_sizes: 边界框尺寸列表
        class_sizes: 按类别划分的边界框尺寸字典
        output_dir: 输出目录路径
    """
    # 转换为numpy数组
    box_sizes_np = np.array(box_sizes)
    
    # 1. 所有目标的尺寸分布散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(box_sizes_np[:, 0], box_sizes_np[:, 1], alpha=0.5, s=10)
    plt.title('Box Size Distribution (All Classes)', fontsize=14)
    plt.xlabel('Normalized Width', fontsize=12)
    plt.ylabel('Normalized Height', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'box_size_scatter.png'), dpi=300)
    plt.close()
    
    # 2. 按类别绘制尺寸分布
    plt.figure(figsize=(15, 12))
    
    for i in range(len(CLASS_NAMES)):
        plt.subplot(4, 3, i+1)
        if class_sizes[i]:
            sizes = np.array(class_sizes[i])
            plt.scatter(sizes[:, 0], sizes[:, 1], alpha=0.5, s=10)
            plt.title(f'{CLASS_NAMES[i]} (n={len(sizes)})', fontsize=10)
        else:
            plt.title(f'{CLASS_NAMES[i]} (n=0)', fontsize=10)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
    
    plt.suptitle('Box Size Distribution by Class', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'box_size_by_class.png'), dpi=300)
    plt.close()
    
    # 3. 尺寸直方图
    plt.figure(figsize=(12, 5))
    
    # 宽度分布
    plt.subplot(1, 2, 1)
    plt.hist(box_sizes_np[:, 0], bins=50, color='skyblue', alpha=0.7)
    plt.title('Width Distribution', fontsize=12)
    plt.xlabel('Normalized Width', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.grid(alpha=0.3)
    
    # 高度分布
    plt.subplot(1, 2, 2)
    plt.hist(box_sizes_np[:, 1], bins=50, color='lightgreen', alpha=0.7)
    plt.title('Height Distribution', fontsize=12)
    plt.xlabel('Normalized Height', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'box_size_histogram.png'), dpi=300)
    plt.close()
    
    print(f"边界框尺寸分布图表已保存到: {output_dir}")

def main():
    """
    主函数：执行数据集可视化
    """
    # 解析命令行参数
    args = parse_args()
    
    # 构建数据集路径
    data_dir = Path(args.data_dir)
    label_dir = data_dir / 'labels' / args.split
    
    # 检查路径是否存在
    if not label_dir.exists():
        print(f"错误：标注目录不存在: {label_dir}")
        print("请先运行visdrone2yolo.py转换数据格式")
        return
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== 开始分析 {args.split} 数据集 ===")
    print(f"数据集路径: {data_dir}")
    print(f"标注文件数量: {len(list(label_dir.glob('*.txt')))}")
    
    # 1. 统计类别分布
    print("\n1. 统计类别分布...")
    class_counts, total_objects = count_classes(label_dir)
    
    # 打印类别统计结果
    print("\n类别统计结果:")
    print("-" * 40)
    print(f"{'类别':<20} {'数量':<10} {'占比':<10}")
    print("-" * 40)
    for i, count in class_counts.items():
        percentage = (count / total_objects) * 100 if total_objects > 0 else 0
        print(f"{CLASS_NAMES[i]:<20} {count:<10} {percentage:.2f}%")
    print("-" * 40)
    print(f"{'总计':<20} {total_objects:<10} 100.00%")
    
    # 绘制类别分布图表
    plot_class_distribution(
        class_counts, 
        total_objects, 
        os.path.join(output_dir, f'class_distribution_{args.split}.png')
    )
    
    # 2. 分析边界框尺寸
    print("\n2. 分析边界框尺寸分布...")
    box_sizes, class_sizes = analyze_box_sizes(label_dir)
    
    # 绘制边界框尺寸分布图表
    plot_box_size_distribution(box_sizes, class_sizes, output_dir)
    
    print(f"\n=== 可视化完成 ===")
    print(f"可视化结果已保存到: {output_dir}")

if __name__ == '__main__':
    main()
