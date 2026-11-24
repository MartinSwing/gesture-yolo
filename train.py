"""用 YOLOv8n 在本机（如 RTX 4060）上训练手势检测模型。

把本文件放到解压后的数据集文件夹中（和 data.yaml 同级），然后运行：
    python train.py

训练完成后，最好的权重在 runs/gesture_yolov8n/weights/best.pt。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from fix_data_yaml import class_names, prepare_data_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 YOLOv8 手势检测模型")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="data.yaml 所在目录或文件路径；默认使用本脚本所在文件夹",
    )
    parser.add_argument("--model", default="yolov8m.pt", help="起始权重，用YOLOv8m模型")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像边长")
    parser.add_argument("--batch", type=int, default=8, help="batch size 8很稳定")
    parser.add_argument(
        "--device",
        default=None,
        help="设备编号，默认自动选择：有 CUDA 用 0，否则用 cpu",
    )
    parser.add_argument("--name", default="gesture_yolov8m", help="本次训练结果文件夹名")
    parser.add_argument("--patience", type=int, default=20, help="验证指标连续多少轮不提升就提前停止")
    return parser.parse_args()


def pick_device(explicit: str | None) -> str:
    """优先使用本机显卡；没有可用 CUDA 时退回 CPU 并给出提示。"""
    if explicit:
        return explicit
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"检测到显卡：{name}，将使用 CUDA:0")
        return "0"
    print("未检测到 CUDA，将使用 CPU 训练（会比较慢）")
    return "cpu"


def main() -> None:
    args = parse_args()
    dataset_dir = args.data if args.data else Path(__file__).resolve().parent
    if dataset_dir.is_file():
        dataset_dir = dataset_dir.parent

    # 修正 Roboflow 导出路径，避免训练时找不到图片
    data_yaml = prepare_data_yaml(dataset_dir)
    names = class_names(data_yaml)
    print(f"数据配置：{data_yaml}")
    print(f"类别：{names}")

    device = pick_device(args.device)
    # yolov8n.pt 会在首次运行时自动下载官方预训练权重
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(dataset_dir / "runs"),
        name=args.name,
        patience=args.patience,
        exist_ok=True,
        pretrained=True,
        plots=True,  # 保存损失曲线和混淆矩阵，写报告时能用上
    )
    best = dataset_dir / "runs" / args.name / "weights" / "best.pt"
    print("训练结束。")
    print(f"最好权重：{best}")
    print("接下来请运行：python val.py")


if __name__ == "__main__":
    main()
