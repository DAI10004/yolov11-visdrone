
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv11训练脚本
功能：使用YOLOv11n模型训练VisDrone2019数据集
"""

from ultralytics import YOLO
import os
import argparse

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='YOLOv11训练VisDrone2019数据集')
    parser.add_argument('--data', type=str, default='configs/VisDrone.yaml', 
                      help='数据集配置文件路径')
    parser.add_argument('--epochs', type=int, default=100, 
                      help='训练轮数')
    parser.add_argument('--batch', type=int, default=8, 
                      help='批次大小')
    # 将第25-26行的默认学习率从0.01改为0.0015
    parser.add_argument('--lr', type=float, default=0.0015, 
                      help='学习率')
    parser.add_argument('--optimizer', type=str, default='sgd', 
                      choices=['sgd', 'adam', 'adamw', 'rmsprop'],
                      help='优化器类型')
    parser.add_argument('--device', type=str, default='0', 
                      help='训练设备，如0或0,1或cpu')
    parser.add_argument('--workers', type=int, default=2, 
                      help='数据加载工作线程数')
    parser.add_argument('--model', type=str, default='weights/best.pt', 
                      help='模型权重文件路径')
    parser.add_argument('--name', type=str, default='visdrone_yolo11', 
                      help='训练任务名称')
    return parser.parse_args()

def main():
    """
    主函数：执行YOLOv11模型训练
    """
    # 解析命令行参数
    args = parse_args()
    
    # 加载预训练模型
    print(f"\n=== 加载模型: {args.model} ===")
    model = YOLO(args.model, task='detect')
    
    # 打印模型信息
    print(f"模型类型: {model.model.__class__.__name__}")
    print(f"输入尺寸: {model.args['imgsz']}")
    
    # 开始训练
    print(f"\n=== 开始训练 ===")
    print(f"数据集配置: {args.data}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch}")
    print(f"学习率: {args.lr}")
    print(f"优化器: {args.optimizer}")
    print(f"训练设备: {args.device}")
    print(f"工作线程: {args.workers}")
    
    try:
        # 执行训练
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            lr0=args.lr,  # 初始学习率
            optimizer=args.optimizer,
            device=args.device,
            workers=args.workers,
            name=args.name,
            project="weights/runs",  # 所有训练成果都放在weights/runs里
            # 数据增强配置（可选，根据需要调整）
            augment=True,
            # 早停机制（可选）
            patience=10,

            # 验证间隔
            val=True
        )
        
        print(f"\n=== 训练完成 ===")
        print(f"训练结果: {results}")
        print(f"最佳模型保存路径: {results.save_dir}/weights/best.pt")
        
    except Exception as e:
        print(f"\n=== 训练失败 ===")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
