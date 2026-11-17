"""从仓库原始CSV重新完成C预处理、伪标签与位置级划分。"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description='一键复现当前已完成的数据处理链路')
    parser.add_argument('--out',type=Path,default=ROOT/'analysis/reproduction',help='新的输出目录，不覆盖已有结果')
    args=parser.parse_args()
    out=args.out.resolve()
    if out.exists():
        raise FileExistsError('输出目录已存在，请用--out指定新的目录')
    # 第1步：先验证下载的数据。默认在仓库根目录运行，所有路径均显式传递。
    def run(script,*arguments):
        command=[sys.executable,'-X','utf8',str(ROOT/'scripts'/script),*map(str,arguments)]
        print('运行：',' '.join(command),flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
    run('verify_repository_data.py')
    # 第2步：只逐谱处理，不从验证集/测试集学习归一化参数。
    raw=ROOT/'datasets/raw'
    run('preprocess_c.py','--data-dir',raw,'--out',out/'C')
    # 第3步：重新计算多峰OPD与局部中心相位伪标签。
    run('generate_pseudo_labels.py','--data-dir',raw,'--inputs',out/'C/model_inputs_C_2048.npz',
        '--out',out/'labels')
    # 第4步：复核10条顺序，按相同位置/重复规则和随机种子重新划分。
    run('split_dataset_by_position.py','--labeled-data',out/'labels/training_C_with_pseudo_labels.npz',
        '--raw-dir',raw,'--out',out/'position_split','--seed',20261117,'--no-raw')
    # 第5步：重建包只保留处理后数组，原始CSV已在仓库datasets/raw中。
    run('verify_dataset_package.py',out/'position_split')
    print('数据链路复现完成：',out,flush=True)


if __name__=='__main__':
    main()
