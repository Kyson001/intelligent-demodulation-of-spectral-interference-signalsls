"""核查仓库中的原始CSV、三份处理后数据、标签及位置分组。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def verify_repository_data(data_root):
    manifest=json.loads((data_root/'manifest.json').read_text(encoding='utf-8'))
    with (data_root/'manifests/sample_manifest.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    mapping={r['sample_id']:r for r in rows}
    if len(mapping)!=4000 or len(rows)!=4000:
        raise AssertionError('逐条清单缺失或sample_id重复')
    raw_files=sorted((data_root/'raw').glob('file*.csv'))
    if {p.stem for p in raw_files}!=set(mapping):
        raise AssertionError('原始CSV文件名/数量与清单不一致')
    for path in raw_files:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=mapping[path.stem]['source_sha256']:
            raise AssertionError(f'原始文件哈希不符：{path.name}')
    ids,positions,duplicates={},{},{}
    counts={}
    for split in ['train','validation','test']:
        path=data_root/f'processed/{split}.npz'
        expected=manifest['processed'][split]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']:
            raise AssertionError(f'{split}文件哈希不符')
        d=np.load(path)
        if d['X'].shape!=(expected['samples'],2048) or not np.isfinite(d['X']).all():
            raise AssertionError(f'{split}输入形状或数值错误')
        if set(d['split'])!={split}:
            raise AssertionError('split字段不符')
        p,n=np.unique(d['position_id'],return_counts=True)
        if not np.all(n==10) or len(p)!=expected['positions']:
            raise AssertionError('位置数或每位置10条不完整')
        if not np.isfinite(d['opd_pseudo_m'][d['opd_valid']]).all():
            raise AssertionError('有效OPD目标存在非有限数')
        good=d['phase_valid']
        np.testing.assert_allclose((d['phase_sin']**2+d['phase_cos']**2)[good],1,atol=1e-10)
        for i,name in enumerate(d['sample_ids']):
            row=mapping[str(name)]
            if row['position_id']!=d['position_id'][i] or row['split']!=split:
                raise AssertionError('数组与逐条清单的位置/分区不一致')
            if row['source_sha256']!=d['source_sha256'][i]:
                raise AssertionError('数组来源哈希与原始数据不一致')
        ids[split]=set(d['sample_ids'])
        positions[split]=set(p)
        duplicates[split]=set(d['duplicate_group'])
        counts[split]=dict(samples=len(d['sample_ids']),positions=len(p))
    for left,right in [('train','validation'),('train','test'),('validation','test')]:
        for groups in [ids,positions,duplicates]:
            if groups[left]&groups[right]:
                raise AssertionError('样本、位置或精确重复谱跨分区泄漏')
    if set.union(*ids.values())!=set(mapping):
        raise AssertionError('三份数据集没有完整覆盖4000条')
    result=dict(raw_csvs=len(raw_files),raw_sha256_verified=True,processed_sha256_verified=True,
                no_position_or_duplicate_leakage=True,splits=counts)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,default=ROOT/'datasets')
    args=parser.parse_args()
    print(json.dumps(verify_repository_data(args.data_root),ensure_ascii=False,indent=2))
