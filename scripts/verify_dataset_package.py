"""核对桌面交付包：数组/标签与源包完全一致，且不存在位置或重复谱泄漏。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def verify_package(folder,verify_raw_hashes=False):
    summary=json.loads((folder/'grouping/split_summary.json').read_text(encoding='utf-8'))
    source=np.load(folder/'source_artifacts/full_C_with_pseudo_labels.npz')
    source_ids=source['sample_ids']
    lookup={str(value):i for i,value in enumerate(source_ids)}
    if len(lookup)!=len(source_ids):
        raise AssertionError('源包有重复sample_id')
    ids,positions,duplicates={}, {}, {}
    report=dict(dataset_arrays_match_source=True,labels_and_masks_match_source=True,
                no_position_leakage=True,no_exact_duplicate_leakage=True,
                raw_hashes_rechecked=verify_raw_hashes,splits={})
    for split in ['train','validation','test']:
        path=folder/f'datasets/{split}/processed_C_with_pseudo_labels.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=summary['dataset_sha256'][split]:
            raise AssertionError(f'{split}文件哈希不匹配')
        part=np.load(path)
        order=np.array([lookup[str(name)] for name in part['sample_ids']])
        for key in ['X','opd_pseudo_m','phase_pseudo_rad','phase_sin','phase_cos','opd_valid','phase_valid','source_sha256']:
            np.testing.assert_equal(part[key],source[key][order])
        np.testing.assert_equal(part['k_rad_per_m'],source['k_rad_per_m'])
        if not np.isfinite(part['X']).all():
            raise AssertionError('网络输入有非有限数')
        if set(part['split'])!={split}:
            raise AssertionError('分区字段不一致')
        position,counts=np.unique(part['position_id'],return_counts=True)
        if not np.all(counts==summary['ordering_audit']['repeats']):
            raise AssertionError('同位置重复数不完整')
        ids[split]=set(part['sample_ids'])
        positions[split]=set(position)
        duplicates[split]=set(part['duplicate_group'])
        raw_files=list((folder/f'datasets/{split}/raw').glob('*.csv'))
        if summary['raw_copied'] and {p.stem for p in raw_files}!=ids[split]:
            raise AssertionError('原始CSV数量或文件名不对应')
        if verify_raw_hashes:
            for i,name in enumerate(part['sample_ids']):
                raw=folder/f'datasets/{split}/raw/{name}.csv'
                if hashlib.sha256(raw.read_bytes()).hexdigest()!=part['source_sha256'][i]:
                    raise AssertionError(f'原始文件损坏：{name}')
        report['splits'][split]=dict(samples=len(part['sample_ids']),positions=len(position),raw_csvs=len(raw_files))
    for left,right in [('train','validation'),('train','test'),('validation','test')]:
        for collection in [ids,positions,duplicates]:
            if collection[left]&collection[right]:
                raise AssertionError('发现跨分区样本/位置/重复泄漏')
    if set.union(*ids.values())!=set(source_ids):
        raise AssertionError('合计样本不等于源包，存在遗漏或多余记录')
    if hashlib.sha256((folder/'source_artifacts/full_C_with_pseudo_labels.npz').read_bytes()).hexdigest()!=summary['source_bundle_sha256']:
        raise AssertionError('源包哈希不匹配')
    report['total_samples']=len(source_ids)
    (folder/'grouping/verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    parser.add_argument('--verify-raw-hashes',action='store_true')
    args=parser.parse_args()
    print(json.dumps(verify_package(args.folder,args.verify_raw_hashes),ensure_ascii=False,indent=2))
