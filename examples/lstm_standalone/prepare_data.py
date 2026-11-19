"""将仓库中已发布的三份NPZ复制到独立工程；已有文件仅校验，不覆盖。"""
import argparse
import hashlib
from pathlib import Path
import shutil


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='为独立LSTM工程准备data目录')
    parser.add_argument('--source', type=Path, default=project.parents[1] / 'datasets' / 'processed')
    args = parser.parse_args()
    destination = project / 'data'
    destination.mkdir(exist_ok=True)
    for name in ('train.npz', 'validation.npz', 'test.npz'):
        source, target = args.source / name, destination / name
        expected = hashlib.sha256(source.read_bytes()).hexdigest()
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError(f'{target}与仓库数据不同，请先核对，脚本不会覆盖')
        else:
            shutil.copy2(source, target)
        print(f'{name}：SHA256校验通过')


if __name__ == '__main__':
    main()
