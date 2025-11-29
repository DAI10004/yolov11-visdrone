#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisDrone2019数据集格式转换脚本
功能：将VisDrone格式标注转换为YOLOv5/YOLOv11格式
"""

import os
from pathlib import Path
import argparse
from PIL import Image
from tqdm import tqdm

def convert_box(size, box):
    """
    将VisDrone格式的边界框转换为YOLO格式
    
    参数：
        size: 图像尺寸 (width, height)
        box: VisDrone格式边界框 [x1, y1, w, h]
            x1, y1: 左上角坐标
            w, h: 宽度和高度
    
    返回：
        YOLO格式边界框 [center_x, center_y, width, height]（归一化后）
    """
    dw = 1.0 / size[0]  # 宽度归一化因子
    dh = 1.0 / size[1]  # 高度归一化因子
    
    # 计算中心点坐标
    center_x = (box[0] + box[2] / 2.0) * dw
    center_y = (box[1] + box[3] / 2.0) * dh
    
    # 归一化宽度和高度
    width = box[2] * dw
    height = box[3] * dh
    
    return center_x, center_y, width, height

def visdrone2yolo(annotations_dir):
    """
    转换指定目录下的VisDrone标注文件为YOLO格式
    
    参数：
        annotations_dir: 标注文件目录路径，格式为 datasets/visdrone2019/annotations/{split}
    """
    # 获取split名称（train, val, test）
    split = annotations_dir.name
    # 获取数据集根目录
    root_dir = annotations_dir.parent.parent
    
    # 创建labels/{split}目录存放YOLO格式标注
    labels_dir = root_dir / 'labels' / split
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 图像文件目录
    images_dir = root_dir / 'images' / split
    
    # 遍历所有标注文件
    pbar = tqdm(annotations_dir.glob('*.txt'), desc=f'Converting {split}')
    for f in pbar:
        # 对应的图像文件路径
        img_path = images_dir / f.name.with_suffix('.jpg')
        
        # 检查图像文件是否存在
        if not img_path.exists():
            print(f"Warning: Image {img_path} not found, skipping annotation {f.name}")
            continue
        
        # 获取图像尺寸
        try:
            img = Image.open(img_path)
            img_size = img.size  # (width, height)
            img.close()
        except Exception as e:
            print(f"Error opening image {img_path}: {e}, skipping annotation {f.name}")
            continue
        
        lines = []
        try:
            with open(f, 'r') as file:
                # 读取并处理每一行标注
                for row in [x.split(',') for x in file.read().strip().splitlines()]:
                    # VisDrone标注格式：[x1, y1, w, h, score, category, truncation, occlusion]
                    if len(row) < 8:
                        continue  # 跳过格式不正确的行
                    
                    # 跳过忽略区域（category=0）
                    if row[5] == '0':
                        continue
                    
                    # 跳过置信度为0的标注
                    if row[4] == '0':
                        continue
                    
                    # 类别号转换：VisDrone类别从1开始，YOLO从0开始
                    cls = int(row[5]) - 1
                    
                    # 确保类别号在有效范围内（0-9，共10个类别）
                    if cls < 0 or cls > 9:
                        continue
                    
                    # 转换边界框
                    box = convert_box(img_size, tuple(map(int, row[:4])))
                    
                    # 确保边界框在有效范围内
                    if all(0 <= x <= 1 for x in box):
                        lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
        except Exception as e:
            print(f"Error processing annotation {f.name}: {e}, skipping")
            continue
        
        # 写入YOLO格式标注文件
        if lines:
            label_path = labels_dir / f.name
            with open(label_path, 'w') as fl:
                fl.writelines(lines)

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Convert VisDrone annotations to YOLO format')
    parser.add_argument('--dir_path', type=str, default='../datasets/visdrone2019', 
                      help='Path to VisDrone dataset root directory')
    args = parser.parse_args()
    
    # 转换指定目录下的训练、验证和测试集
    dir = Path(args.dir_path)
    for split in ['train', 'val', 'test']:
        # 检查annotations目录是否存在
        annotations_dir = dir / 'annotations' / split
        if annotations_dir.exists():
            visdrone2yolo(annotations_dir)
