"""用笔记本摄像头自动连拍爱心图：空格开始/暂停，q 退出。"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="heart_solo")
    parser.add_argument("--tag", default="far", help="场景名：far / face / high")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.4, help="自动保存间隔（秒）")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / args.out / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError("打不开摄像头，可把 --cam 改成 1 再试")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    saved = 0
    recording = False
    last_save = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        if recording and now - last_save >= args.interval:
            name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            path = out_dir / f"{args.tag}_{name}.jpg"
            cv2.imwrite(str(path), frame)
            saved += 1
            last_save = now
            print(f"已保存 {path}")

        view = frame.copy()
        status = "RECORDING" if recording else "PAUSED"
        cv2.putText(
            view,
            f"{args.tag}  {status}  saved:{saved}  SPACE=start/pause  q=quit",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if recording else (0, 255, 255),
            2,
        )
        cv2.imshow("capture_heart", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            recording = not recording
            last_save = 0.0
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"共保存 {saved} 张，目录：{out_dir}")

if __name__ == "__main__":
    main()