# 实时手势检测（YOLOv8）

用 YOLOv8m 在摄像头画面中检测 **Paper / Rock / Scissors / heart（双手比心）**。本仓库只包含训练、验证和实时检测脚本，**不含**训练集图片、标签和模型权重。

本项目是岭南大学 Data Science 硕士课程 CDS540 Computer Vision 小组课设的检测部分：在组员分类 CNN 方案之外，改用目标检测，以便同时给出手势类别和位置，并更好地适应手离镜头远近、举高、部分遮挡等情况。

## 功能

- 四类手势检测：Paper、Rock、Scissors、heart
- 支持从预训练权重训练，以及在已有权重上继续微调
- 实时摄像头检测（`recognize_live.py`）
- 附带一个人即可完成的爱心采集脚本（定时连拍，双手比心时不必按键保存）

## 环境

建议使用已安装 CUDA 版 PyTorch 的 Python 环境（本机训练时为 Python 3.11、PyTorch 2.5.1 + CUDA 12.1、Ultralytics 8.4.x，NVIDIA RTX 4060）。

```bash
pip install -r requirements.txt
```

主要依赖：`ultralytics`、`opencv-python`、`torch`。若已有可用的 CUDA 版 PyTorch，不必再装 CPU 版。

## 仓库里有什么、没有什么

**包含：** 训练 / 验证 / 微调 / 实时检测脚本、数据采集脚本、`data.yaml` 示例、本说明。

**不包含（请不要把它们提交到公开仓库）：**

- `train/`、`valid/`、`test/` 中的图像和标签
- `runs/` 中的训练日志与 `.pt` 权重
- 自拍原始图片（如 `heart_solo/`）
- 官方预训练文件 `yolov8m.pt`（运行时会自动下载或请自行准备）

自拍数据含人脸与室内背景，属于个人数据；SXSW、heart-4mxim 等公开集请按原协议自行下载，不要二次打包分发。

## 数据准备

请在本仓库根目录自行建立 YOLO 目录结构：

```text
train/images  train/labels
valid/images  valid/labels
test/images   test/labels
```

类别约定（`nc: 4`）：

| 编号 | 名称 |
| --- | --- |
| 0 | Paper |
| 1 | Rock |
| 2 | Scissors |
| 3 | heart |

### 1. 猜拳主数据集（SXSW）

使用 Roboflow 上的石头剪刀布检测集（YOLOv8 格式，仅 Paper / Rock / Scissors）。下载并解压后，将 `train`、`valid`、`test` 中的图像和标签放入上述目录。

### 2. 公开爱心补充集（heart-4mxim）

来源：[mahdi-marani/heart-4mxim](https://universe.roboflow.com/mahdi-marani/heart-4mxim)（v1）。该集原始类别编号为 `0`，合并前必须把每个标签文件行首的 `0` 改成 `3`，否则会和 Paper 冲突。请批量把每个标签文件行首的类别号从 `0` 改成 `3`，再复制进同一套 `train/valid/test` 目录，**只添加、不覆盖**已有猜拳文件。

说明：该公开集样本较少，且含有不太适合手势语义的增强（如垂直翻转、90° 旋转），不能单独解决真实摄像头中的爱心漏检。

### 3. 自建爱心补充集（建议）

公开爱心图与笔记本摄像头域差异较大。更有效的做法是用同一台演示用摄像头补拍，尤其覆盖这些容易漏检的场景：

- 手离镜头较远、爱心在画面中较小
- 双手举高、出现在画面偏上
- 心形中间露出脸

可使用 `capture_heart.py`：先摆好姿势，按空格开始定时连拍，按 `q` 退出。

```bash
python capture_heart.py --out heart_solo --tag far --interval 0.4
```

`--tag` 可改为 `face` 或 `high`。采集后需要打上 YOLO 检测框（类别为 `3`），再划分进 `train/valid/test`，同样只添加、不覆盖。请勿将未处理的自拍上传到公开仓库。

准备好数据后，将 `data_local.yaml` 中的 `path` 改成你本机的数据集根目录（不要使用他人的绝对路径）。`data.yaml` 仅作类别与相对路径示例。

## 训练

请在数据根目录（与本仓库脚本放在一起时即当前目录）运行。Windows 上请设置 `workers=0`，并把训练入口放在 `if __name__ == "__main__":` 中，否则可能出现多进程报错。

### 第一步：在猜拳数据上训练 YOLOv8m

```bash
python train.py
```

默认使用 `yolov8m.pt`，例如 `epochs=50`、`imgsz=640`、`batch=8`。输出目录示例：`runs/gesture_yolov8m/weights/best.pt`。

### 第二步：合并爱心数据后微调

不要只用爱心数据从头训练，否则石头剪刀布会被忘掉。应从上一步的 `best.pt` 继续：

```bash
python finetune_heart.py
```

请在脚本中确认：

- 起始权重指向已有的 `best.pt`（例如第一轮猜拳模型，或上一轮爱心微调模型）
- `name` 使用新的运行名（如 `gesture_yolov8m_heart_self2`），避免覆盖上一版权重

### 第三步：测试集评估

```bash
python val.py --split test --weights runs/gesture_yolov8m_heart_self2/weights/best.pt
```

`val.py` 与 `recognize_live.py` 默认不会自动找到你的微调权重，评估和演示时都要加 `--weights`。

本项目一次完整实验的测试集结果（387 张，含自建爱心测试图）供参考：

| 类别 | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| all | 0.962 | 0.749 |
| Paper | 0.927 | 0.718 |
| Rock | 0.970 | 0.777 |
| Scissors | 0.954 | 0.800 |
| heart | 0.995 | 0.700 |

其中爱心在仅用公开补充集时，测试 mAP50-95 约为 0.436；加入自建远景 / 举高补充并微调后，提升至 0.700。不同数据划分和训练设置下，结果会有差异。

## 实时检测

```bash
python recognize_live.py --weights runs/gesture_yolov8m_heart_self2/weights/best.pt
```

请把路径换成你本地实际的 `best.pt`。按 `q` 退出。

## 致谢

- 石头剪刀布检测数据：SXSW / Roboflow 发布的 YOLOv8 格式公开集
- 公开爱心检测数据：[heart-4mxim](https://universe.roboflow.com/mahdi-marani/heart-4mxim)（Mahdi Marani）
- 检测模型：[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- 部分爱心样本为作者用笔记本摄像头自建并标注，仅用于课程实验，默认不随仓库发布

如将本项目用于课程报告或论文，请一并致谢上述来源。

## 许可

脚本按仓库中的许可证使用（若未另行说明，仅供学习与课程交流）。第三方数据集和 YOLOv8 权重遵循其各自的许可协议，使用前请自行查阅。
