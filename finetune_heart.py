"""用已有的 YOLOv8m best.pt，在合并爱心后的数据上微调。"""
from pathlib import Path
from ultralytics import YOLO

def main():
    root = Path(__file__).resolve().parent
    data = root / "data_local.yaml"
    weights = root / "runs" / "gesture_yolov8m" / "weights" / "best.pt"

    if not data.exists():
        raise FileNotFoundError(f"找不到数据配置: {data}")
    if not weights.exists():
        raise FileNotFoundError(f"找不到起始权重: {weights}")

    model = YOLO(str(weights))
    model.train(
        data=str(data),
        epochs=50,   # 微调的轮数
        imgsz=640,   # 图像大小
        batch=8,     # 批处理大小
        device=0,    # 使用的设备，0表示GPU，-1表示CPU
        workers=0,   # 数据加载的线程数
        project=str(root / "runs"),    # 保存路径
        name="gesture_yolov8m_heart_self2",  # 保存路径
        patience=15,                   # 提前终止的耐心
        exist_ok=True,                 # 如果目录存在，是否允许覆盖
        pretrained=True,               # 是否使用预训练权重
        plots=True,                    # 是否生成图表
    )

    best = root / "runs" / "gesture_yolov8m_heart_self" / "weights" / "best.pt"
    print("微调结束。")
    print(f"最好权重: {best}")
    print("接下来请运行: python val.py --split test --weights runs/gesture_yolov8m_heart/weights/best.pt")

if __name__ == "__main__":
    main()