from ultralytics import YOLO
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv11推理脚本')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='模型权重路径')
    parser.add_argument('--source', type=str, required=True, help='输入图片/视频/目录路径')
    parser.add_argument('--save', action='store_true', default=True, help='保存推理结果')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    return parser.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.weights)
    results = model(args.source, conf=args.conf, save=args.save)
    print(f"推理完成，结果保存在：{results[0].save_dir}")

if __name__ == '__main__':
    main()
