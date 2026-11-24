"""检查并修正 data.yaml 里的路径，让训练脚本能在本机找到图片。"""

from __future__ import annotations

from pathlib import Path

import yaml


def _first_existing(root: Path, candidates: list[str]) -> str | None:
    """在数据集根目录下，返回第一个真实存在的相对路径。"""
    for relative in candidates:
        if (root / relative).exists():
            return relative
    return None


def prepare_data_yaml(dataset_dir: Path) -> Path:
    """读取解压后的 data.yaml，把路径改成相对本机数据集根目录，并另存为 data_local.yaml。

    Roboflow 导出的 yaml 经常带有 ../ 或云端路径，直接训会找不到图片。
    这个函数不覆盖原始 data.yaml，方便以后对照。
    """
    dataset_dir = dataset_dir.resolve()
    source = dataset_dir / "data.yaml"
    if not source.exists():
        raise FileNotFoundError(
            f"未找到 {source}。请把本脚本放到解压后的数据集文件夹里，"
            "或用 --data 指定 data.yaml 的位置。"
        )

    with source.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    train = _first_existing(dataset_dir, ["train/images", "train"])
    valid = _first_existing(dataset_dir, ["valid/images", "valid", "val/images", "val"])
    test = _first_existing(dataset_dir, ["test/images", "test"])
    if train is None or valid is None:
        raise FileNotFoundError(
            f"在 {dataset_dir} 下没有找到 train/valid 图片文件夹，请检查是否解压完整。"
        )

    # path 写成数据集根目录的绝对路径，避免从其他工作目录启动时找不到图
    config["path"] = str(dataset_dir)
    config["train"] = train
    config["val"] = valid
    if test:
        config["test"] = test

    output = dataset_dir / "data_local.yaml"
    with output.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    return output


def class_names(data_yaml: Path) -> list[str]:
    """读取 yaml 中的类别名称，供打印和实时检测脚本使用。"""
    with data_yaml.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    names = config.get("names", [])
    if isinstance(names, dict):
        return [names[key] for key in sorted(names, key=lambda item: int(item))]
    return list(names)
