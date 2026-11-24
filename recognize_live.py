"""打开摄像头，用训练好的 YOLOv8 权重做实时手势检测。

先完成训练得到 best.pt，再运行：
    python recognize_live.py

按 q 或 Esc 退出。窗口可以拖角落缩放。
当前公开数据集只有石头 / 剪刀 / 布；双手比心需要以后补数据再训练。
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# 英文类别（与 data.yaml 一致）到中文显示名
CHINESE_NAMES = {
    "Rock": "石头（握拳）",
    "Paper": "布（张掌）",
    "Scissors": "剪刀",
    "Love": "爱心",
    "heart": "爱心",
    "rock": "石头（握拳）",
    "paper": "布（张掌）",
    "scissors": "剪刀",
}

# 可选：打开 --control 后，稳定识别到某手势时发送的按键（可按小组方案自行改）
# 这里只是示例，默认不发送按键，避免上课或写报告时误触
GESTURE_KEYS = {
    "Paper": "space",      # 张掌：播放 / 暂停
    "Rock": "down",        # 握拳：下一个（示例）
    "Scissors": "right",   # 剪刀：下一页（示例）
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 实时手势检测")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="权重路径；默认使用脚本同目录下 runs/gesture_yolov8n/weights/best.pt",
    )
    parser.add_argument("--name", default="gesture_yolov8n", help="训练结果文件夹名")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS 的 IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸，需与训练接近")
    parser.add_argument("--device", default=None, help="设备编号，默认自动选择")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号")
    parser.add_argument("--history", type=int, default=8, help="多帧投票的窗口长度，减少标签闪烁")
    parser.add_argument(
        "--control",
        action="store_true",
        help="打开后，稳定识别到的手势会发送键盘按键（需安装 pyautogui）",
    )
    return parser.parse_args()


def pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    return "0" if torch.cuda.is_available() else "cpu"


def find_weights(script_dir: Path, args: argparse.Namespace) -> Path:
    if args.weights:
        weights = args.weights
    else:
        weights = script_dir / "runs" / args.name / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(
            f"未找到权重 {weights}。请先运行 python train.py，或用 --weights 指定 .pt 文件。"
        )
    return weights


def open_camera(index: int) -> cv2.VideoCapture:
    """Windows 上优先使用 DirectShow，避免打开摄像头很慢或失败。"""
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"无法打开摄像头 {index}，可换一个 --camera 编号再试。")
    return cap


def find_chinese_font(size: int = 32) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """在 Windows / Linux 上找一个能显示中文的字体。"""
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyh.ttf"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


def put_chinese(
    image_bgr: np.ndarray,
    text: str,
    xy: tuple[int, int],
    color: tuple[int, int, int] = (0, 255, 0),
    size: int = 32,
) -> np.ndarray:
    """在 OpenCV 画面上写中文（OpenCV 默认字体不含中文，需借用 Pillow）。"""
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = find_chinese_font(size)
    image_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_rgb)
    # Pillow 用 RGB，OpenCV 传入的是 BGR，这里要交换通道
    rgb = (color[2], color[1], color[0])
    draw.text(xy, text, font=_FONT_CACHE[size], fill=rgb)
    return cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)


def display_name(raw: str) -> str:
    return CHINESE_NAMES.get(raw, raw)


def majority_label(history: deque[str]) -> str | None:
    """对最近几帧的类别做多数投票，得到更稳定的标签。"""
    if not history:
        return None
    counts: dict[str, int] = {}
    for label in history:
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)


def maybe_send_key(label: str, last_sent: str | None) -> str | None:
    """仅在 --control 且标签变化时发送一次按键，避免每一帧都重复触发。"""
    try:
        import pyautogui
    except ImportError:
        print("未安装 pyautogui，无法发送按键。可运行：pip install pyautogui")
        return last_sent
    key = GESTURE_KEYS.get(label)
    if key and label != last_sent:
        pyautogui.press(key)
        print(f"已发送按键：{label} -> {key}")
        return label
    return last_sent


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    weights = find_weights(script_dir, args)
    device = pick_device(args.device)
    print(f"权重：{weights}")
    print(f"设备：{device}")

    model = YOLO(str(weights))
    cap = open_camera(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "YOLOv8 Gesture Demo"
    # WINDOW_NORMAL 后可以用鼠标拖窗口角落缩放
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    history: deque[str] = deque(maxlen=args.history)
    last_sent: str | None = None
    print("检测已启动。请把手放到镜头前。按 q 退出。")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取摄像头失败，退出。")
            break

        # 镜像画面，让屏幕里的手和自己的左右方向一致
        frame = cv2.flip(frame, 1)
        results = model.predict(
            frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
        )[0]

        # 本帧分数最高的一个手势，避免同时画出很多互相打架的框
        best_label = None
        best_conf = 0.0
        best_box = None
        if results.boxes is not None:
            for box in results.boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = results.names[class_id]
                if confidence > best_conf:
                    best_conf = confidence
                    best_label = label
                    best_box = box.xyxy[0].tolist()

        if best_label is not None and best_box is not None:
            history.append(best_label)
            x1, y1, x2, y2 = [int(v) for v in best_box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            caption = f"{display_name(best_label)}  {best_conf:.0%}"
            frame = put_chinese(frame, caption, (x1, max(10, y1 - 40)), size=32)
        else:
            history.clear()

        stable = majority_label(history)
        title = display_name(stable) if stable else "未检测到手势"
        frame = put_chinese(frame, title, (24, 24), color=(0, 255, 0), size=40)
        frame = put_chinese(frame, "按 q 退出", (24, frame.shape[0] - 56), color=(200, 200, 200), size=24)

        if args.control and stable:
            last_sent = maybe_send_key(stable, last_sent)
        elif not stable:
            last_sent = None

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
