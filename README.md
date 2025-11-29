# yolov11-visdrone
建立baseline的训练结果
使用官方预训练权重yolo11n.pt、官方配置yolo11n.yaml、visdrone.yaml
脚本功能详细说明 ：
- visdrone2yolo.py ：数据格式转换，将VisDrone原始标注转为YOLO格式
- data_visualization.py ：数据集可视化，生成类别分布和边界框尺寸分析图表
- train.py ：模型训练，支持自定义参数和多种优化器
- inference.py ：模型推理，支持多种输入类型和批量处理， 是训练结束后的推理工具 ，用于使用训练好的模型处理新数据。训练途中的验证由训练脚本自动完成，不需要单独运行该脚本。
