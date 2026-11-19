"""绘制数据集中单个样本的光谱。用法：python scripts/plot_spectrum.py 31"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="绘制指定编号样本的波长-强度曲线")
    parser.add_argument("sample", type=int, nargs="?", default=31,
                        help="样本编号，例如 31 对应 file0031.csv（默认 31）")
    parser.add_argument("--save", type=Path, help="可选：将图像保存到指定路径")
    parser.add_argument("--no-show", action="store_true", help="不弹出图窗")
    args = parser.parse_args()
    path = ROOT / "analysis" / "data_4000" / f"file{args.sample:04d}.csv"
    if not path.is_file():
        parser.error(f"找不到样本文件：{path}")
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if data.shape[1] != 2 or not np.isfinite(data).all():
        parser.error(f"文件应包含两列有限数值：{path}")
    wavelength, intensity = data.T
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wavelength, intensity, linewidth=0.8)
    ax.set(xlabel="Wavelength (nm)", ylabel="Intensity (dBm; CSV header)",
           title=f"Spectrum — {path.name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"Saved: {args.save.resolve()}")
    print(f"File: {path}\nPoints: {len(wavelength)}; wavelength: {wavelength.min():g}–{wavelength.max():g} nm")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
