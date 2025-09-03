#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# 树状结构绘制用字符
SPACE = "    "
BRANCH = "│   "
TEE = "├── "
LAST = "└── "

def tree(path: Path, prefix: str = ""):
    try:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        print(prefix + "PermissionError: " + path.name)
        return

    entries_count = len(entries)
    for idx, entry in enumerate(entries):
        connector = LAST if idx == entries_count - 1 else TEE
        print(f"{prefix}{connector}{entry.name}")
        if entry.is_dir():
            extension = SPACE if connector == LAST else BRANCH
            yield from tree(entry, prefix + extension)

def main():
    train_path = os.environ.get('TRAIN_DATA_PATH')
    if not train_path:
        print("ERROR: 环境变量 TRAIN_DATA_PATH 未设置")
        sys.exit(1)

    root = Path(train_path)
    if not root.is_dir():
        print(f"ERROR: TRAIN_DATA_PATH='{train_path}' 不是一个有效目录")
        sys.exit(1)

    print(root.name or root)  # 打印根目录名称或路径
    yield from tree(root)

if __name__ == "__main__":
    for line in main():
        print(line)
