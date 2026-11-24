"""用验证集 / 测试集评估训练得到的 best.pt。

先完成 python train.py，再运行：
    python val.py
    python val.py --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from fix_data_yaml import class_names, prepare_data_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 YOLOv8 手势检测模型")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="数据集文件夹或 data.yaml 路径；默认使用本脚本所在文件夹",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="权重路径；默认使用 runs/gesture_yolov8n/weights/best.pt",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像边长，需与训练时一致")
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="评估哪一个划分：val 为验证集，test 为测试集",
    )
    parser.add_argument("--device", default=None, help="设备编号，默认自动选择")
    parser.add_argument("--name", default="gesture_yolov8n", help="训练结果文件夹名，需与 train.py 一致")
    return parser.parse_args()


def pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    return "0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    args = parse_args()
    dataset_dir = args.data if args.data else Path(__file__).resolve().parent
    if dataset_dir.is_file():
        dataset_dir = dataset_dir.parent

    data_yaml = prepare_data_yaml(dataset_dir)
    weights = args.weights or (dataset_dir / "runs" / args.name / "weights" / "best.pt")
    if not weights.exists():
        raise FileNotFoundError(
            f"未找到权重 {weights}。请先运行 python train.py，或用 --weights 指定 .pt 文件。"
        )

    print(f"权重：{weights}")
    print(f"数据配置：{data_yaml}")
    print(f"类别：{class_names(data_yaml)}")
    print(f"评估划分：{args.split}")

    model = YOLO(str(weights))
    model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        device=pick_device(args.device),
        plots=True,
    )
    print("验证结束。指标图一般在 runs 目录对应的 val 文件夹里。")
    print("若指标看起来正常，可运行：python recognize_live.py")


if __name__ == "__main__":
    main()
