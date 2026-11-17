"""检查每位置连续10条的排列，并导出桌面上的完整分组数据包。

中文步骤：核查顺序 → 检验10种分组偏移 → 形成位置清单 → 绑定跨位置重复
        → 按位置70/15/15划分 → 导出NPZ/CSV/原始光谱/代码 → 检验没有组泄漏。
不以名义10 μm步长生成标签，不删除待复核记录，不修改原始数据。
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def write_csv(path,rows,fields=None):
    """CSV使用UTF-8 BOM，便于Windows Excel直接显示中文。"""
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or (list(rows[0]) if rows else ['note']))
        writer.writeheader(); writer.writerows(rows)


# ==================== 第1块：量化核查“每位置连续10条” ====================
def audit_ordering(data,repeats=10):
    ids=data['sample_ids']
    if not np.array_equal(ids,[f'file{i:04}' for i in range(1,len(ids)+1)]):
        raise ValueError('文件编号缺失、重复或未按连续顺序排列，请先核对采集清单')
    if len(ids)%repeats:
        raise ValueError('总条数不是每位置重复数的整数倍')
    opd=data['opd_pseudo_m']
    if not np.isfinite(opd).all():
        raise ValueError('顺序诊断存在缺失OPD，不能直接使用本次完整分块检验')
    difference=np.abs(np.diff(opd))*1e6
    offsets=[]
    for offset in range(repeats):
        boundary=np.arange(len(difference))%repeats==(offset+repeats-1)%repeats
        within=float(np.median(difference[~boundary]))
        between=float(np.median(difference[boundary]))
        offsets.append(dict(offset=offset,within_median_um=within,boundary_median_um=between,
                            boundary_p10_um=float(np.quantile(difference[boundary],.1)),
                            separation_ratio=between/max(within,1e-15)))
    best=max(offsets,key=lambda row:row['separation_ratio'])
    # 预设门槛：0偏移最强、边界差至少是组内差的5倍，且明显优于其余偏移。
    competitors=max(r['separation_ratio'] for r in offsets if r['offset']!=0)
    supported=(best['offset']==0 and offsets[0]['separation_ratio']>=5 and
               offsets[0]['separation_ratio']>=3*competitors)
    if not supported:
        raise ValueError('数据没有充分支持从第一条起按10条分块；暂停自动位置划分')
    # 第二种诊断：FFT幅度形状，弱化单纯相位平移造成的相关性变化。
    magnitude=np.abs(np.fft.rfft(data['X'],axis=1))
    magnitude[:,0]=0
    magnitude/=np.maximum(np.linalg.norm(magnitude,axis=1,keepdims=True),1e-15)
    distance=np.maximum(0,1-np.sum(magnitude[:-1]*magnitude[1:],axis=1))
    boundary=np.arange(len(distance))%repeats==repeats-1
    groups=opd.reshape(-1,repeats)*1e6
    centers=np.median(groups,axis=1)
    deviations=np.abs(opd*1e6-np.repeat(centers,repeats))
    step=np.diff(centers)
    audit=dict(samples=len(ids),positions=len(centers),repeats=repeats,
               conclusion='strongly_supports_contiguous_blocks_from_file0001',
               evidence_scope='inferred_from_acquisition_description_and_spectra_not_hardware_position_log',
               best_offset=best['offset'],within_opd_median_um=offsets[0]['within_median_um'],
               boundary_opd_median_um=offsets[0]['boundary_median_um'],
               boundary_to_within_ratio=offsets[0]['separation_ratio'],
               fft_shape_within_median=float(np.median(distance[~boundary])),
               fft_shape_boundary_median=float(np.median(distance[boundary])),
               position_median_opd_step_um_p05_p50_p95=np.quantile(step,[.05,.5,.95]).tolist(),
               position_review_rule='absolute deviation from position median OPD > 5 um; flag only',
               position_review_count=int(np.sum(deviations>5)))
    return audit,offsets,centers,deviations


# ==================== 第2块：重复谱所涉及的位置必须绑定 ====================
def position_components(positions,duplicate_groups):
    """并查集：同位置不可拆分；完全重复谱即使跨位置也不得跨分区。"""
    n=int(positions.max())
    parent=list(range(n+1))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:
            parent[max(a,b)]=min(a,b)
    duplicates={}
    for i,key in enumerate(duplicate_groups):
        duplicates.setdefault(str(key),[]).append(i)
    cross=[]
    for key,indices in duplicates.items():
        group_positions=sorted(set(int(positions[i]) for i in indices))
        if len(group_positions)>1:
            for p in group_positions[1:]:
                union(group_positions[0],p)
            cross.append(dict(duplicate_group=key,indices=indices,positions=group_positions))
    components={}
    for p in range(1,n+1):
        components.setdefault(find(p),[]).append(p)
    return list(components.values()),cross


# ==================== 第3块：按位置数精确分配70%/15%/15% ====================
def choose_components(components,target,rng):
    """确定性随机打乱后做子集和，避免拆开大小为2等重复绑定组件。"""
    order=rng.permutation(len(components))
    reachable={0:[]}
    for index in order:
        size=len(components[int(index)])
        for total,chosen in list(reachable.items())[::-1]:
            new=total+size
            if new<=target and new not in reachable:
                reachable[new]=chosen+[int(index)]
        if target in reachable:
            break
    if target not in reachable:
        raise ValueError('无法在不拆开重复绑定位置的条件下达到目标比例')
    selected=set(reachable[target])
    return [c for i,c in enumerate(components) if i in selected],[c for i,c in enumerate(components) if i not in selected]


def assign_positions(components,position_count,seed=20261117):
    """固定随机种子，分配仅使用位置/重复关系，不按标签误差挑选测试集。"""
    rng=np.random.default_rng(seed)
    n_val=round(.15*position_count); n_test=round(.15*position_count)
    validation,remaining=choose_components(components,n_val,rng)
    test,train=choose_components(remaining,n_test,rng)
    mapping={}
    component_ids={}
    for split,groups in [('train',train),('validation',validation),('test',test)]:
        for component in groups:
            key=f'G{min(component):04}'
            for position in component:
                mapping[position]=split; component_ids[position]=key
    if len(mapping)!=position_count:
        raise AssertionError('位置没有完全分配')
    return mapping,component_ids


# ==================== 第4块：汇出训练/验证/测试文件，并保留原始谱 ====================
def build_package(labeled_path,raw_dir,out,repeats=10,seed=20261117,include_raw=True):
    """out必须为新的目录。复制源文件，不做移动或删除，也不使用共享硬链接。"""
    if out.exists():
        raise FileExistsError(f'输出目录已存在，请选择新目录：{out}')
    with np.load(labeled_path) as stream:
        data={key:stream[key] for key in stream.files}
    count=len(data['sample_ids'])
    if data['X'].shape!=(count,2048):
        raise ValueError('本次交付要求X为样本数×2048')
    audit,offsets,centers,deviations=audit_ordering(data,repeats)
    positions=np.arange(count)//repeats+1
    components,cross=position_components(positions,data['duplicate_group'])
    mapping,component_ids=assign_positions(components,len(centers),seed)
    splits=np.array([mapping[int(p)] for p in positions])

    # 导出前验证三个分区间没有位置或完全重复谱相交。
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        for field in [positions,data['duplicate_group']]:
            if set(field[splits==a])&set(field[splits==b]):
                raise AssertionError('发现跨分区位置或重复谱泄漏')
    out.mkdir(parents=True)
    grouping=out/'grouping'; grouping.mkdir()
    rows=[]
    for i,sample in enumerate(data['sample_ids']):
        p=int(positions[i]); split=str(splits[i])
        rows.append(dict(sample_id=str(sample),position_id=f'P{p:04}',position_index=p,
                         repeat_index=i%repeats+1,leakage_component=component_ids[p],split=split,
                         source_sha256=str(data['source_sha256'][i]),
                         duplicate_group=str(data['duplicate_group'][i]),
                         opd_pseudo_m=float(data['opd_pseudo_m'][i]),phase_pseudo_rad=float(data['phase_pseudo_rad'][i]),
                         opd_valid=bool(data['opd_valid'][i]),phase_valid=bool(data['phase_valid'][i]),
                         position_median_opd_um=float(centers[p-1]),
                         deviation_from_position_median_um=float(deviations[i]),
                         position_review=bool(deviations[i]>5),
                         raw_relative_path=f'datasets/{split}/raw/{sample}.csv' if include_raw else 'not_copied'))
    write_csv(grouping/'sample_manifest.csv',rows)
    position_rows=[]
    for p in range(1,len(centers)+1):
        selected=positions==p
        position_rows.append(dict(position_id=f'P{p:04}',first_file=str(data['sample_ids'][np.flatnonzero(selected)[0]]),
                                  last_file=str(data['sample_ids'][np.flatnonzero(selected)[-1]]),
                                  sample_count=int(selected.sum()),split=mapping[p],leakage_component=component_ids[p],
                                  median_opd_um=float(centers[p-1]),
                                  opd_range_um=float(np.ptp(data['opd_pseudo_m'][selected])*1e6),
                                  position_review_count=int(np.sum(deviations[selected]>5))))
    write_csv(grouping/'position_manifest.csv',position_rows)
    write_csv(grouping/'offset_comparison.csv',offsets)
    write_csv(grouping/'position_needs_review.csv',[r for r in rows if r['position_review']],list(rows[0]))
    write_csv(grouping/'pseudo_label_needs_review.csv',[r for r in rows if not r['opd_valid'] or not r['phase_valid']],list(rows[0]))
    write_csv(grouping/'cross_position_duplicates.csv',
              [dict(samples=';'.join(str(data['sample_ids'][i]) for i in c['indices']),
                    positions=';'.join(f'P{p:04}' for p in c['positions']),
                    split=mapping[c['positions'][0]]) for c in cross],['samples','positions','split'])
    audit.update(cross_position_duplicate_groups=len(cross),leakage_components=len(components))
    (grouping/'ordering_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')

    summary=dict(seed=seed,split_method='random_by_position_70_15_15_with_duplicate_components',
                 assignment_basis='positions_and_exact_duplicates_only_not_pseudo_label_values',
                 ordering_audit=audit,raw_copied=include_raw,exclusions=0,
                 position_or_duplicate_leakage=False,
                 ground_truth_status='OPD and center phase are algorithmic pseudo labels',splits={})
    dataset_hashes={}
    for split in ['train','validation','test']:
        folder=out/'datasets'/split; folder.mkdir(parents=True)
        selected=np.flatnonzero(splits==split)
        # 保留所有原有字段、真伪标签声明和任务掩码；不把待复核值改成零。
        subset={key:value[selected] if value.ndim>0 and value.shape[0]==count else value
                for key,value in data.items()}
        subset.update(position_id=np.array([f'P{p:04}' for p in positions[selected]]),
                      position_index=positions[selected],repeat_index=selected%repeats+1,
                      split=splits[selected],position_review=deviations[selected]>5,
                      leakage_component=np.array([component_ids[int(p)] for p in positions[selected]]))
        dataset_path=folder/'processed_C_with_pseudo_labels.npz'
        np.savez_compressed(dataset_path,**subset)
        dataset_hashes[split]=hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        write_csv(folder/'samples_and_labels.csv',[rows[i] for i in selected])
        summary['splits'][split]=dict(samples=len(selected),positions=len(set(positions[selected])),
                                     opd_valid=int(subset['opd_valid'].sum()),phase_valid=int(subset['phase_valid'].sum()),
                                     position_review=int(subset['position_review'].sum()),
                                     opd_range_mm=[float(subset['opd_pseudo_m'].min()*1e3),float(subset['opd_pseudo_m'].max()*1e3)])
        if include_raw:
            target=folder/'raw'; target.mkdir()
            for i in selected:
                source=raw_dir/f'{data["sample_ids"][i]}.csv'
                if hashlib.sha256(source.read_bytes()).hexdigest()!=str(data['source_sha256'][i]):
                    raise ValueError(f'原始文件已改变，停止复制：{source.name}')
                destination=target/source.name
                shutil.copyfile(source,destination)
                if hashlib.sha256(destination.read_bytes()).hexdigest()!=str(data['source_sha256'][i]):
                    raise IOError(f'复制内容校验失败：{destination.name}')
        print(f'{split}: {len(selected)}条、{summary["splits"][split]["positions"]}个位置，导出完成',flush=True)

    # 报告随机留出中是否存在训练OPD范围外的边界样本，不事后重新挑选测试集。
    train_opd=data['opd_pseudo_m'][splits=='train']
    for split in ['train','validation','test']:
        values=data['opd_pseudo_m'][splits==split]
        summary['splits'][split]['outside_train_opd_range']=int(np.sum((values<train_opd.min())|(values>train_opd.max())))

    # ==================== 第5块：将预处理、伪标签和划分代码一起打包 ====================
    code=out/'code'; code.mkdir()
    sources=['scripts/preprocess_c.py','scripts/generate_pseudo_labels.py','scripts/split_dataset_by_position.py',
             'src/demodulation/__init__.py','src/demodulation/preprocessing.py','src/demodulation/pseudo_labels.py',
             'configs/preprocessing_comparison.json','configs/training_preprocessing.json','configs/pseudo_labels.json',
             'tests/test_preprocess_c.py','tests/test_preprocessing.py','tests/test_pseudo_labels.py',
             'tests/test_position_split.py','scripts/verify_dataset_package.py','requirements.txt']
    for relative in sources:
        source=ROOT/relative
        if source.exists():
            target=code/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,target)
    docs=out/'docs'; docs.mkdir()
    for source in (ROOT/'docs').glob('0[4567]_*.md'):
        shutil.copyfile(source,docs/source.name)
    source_bundle=out/'source_artifacts'; source_bundle.mkdir()
    shutil.copyfile(labeled_path,source_bundle/'full_C_with_pseudo_labels.npz')
    for name in ['pseudo_labels.csv','needs_review.csv','summary.json']:
        if (labeled_path.parent/name).exists():
            shutil.copyfile(labeled_path.parent/name,source_bundle/name)
    summary['source_bundle_sha256']=hashlib.sha256(labeled_path.read_bytes()).hexdigest()
    summary['dataset_sha256']=dataset_hashes
    (grouping/'split_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'README_使用说明.md').write_text(package_readme(summary),encoding='utf-8')
    return summary


def package_readme(summary):
    """生成面向使用者的说明，不把数据分区包装成独立仪器验证。"""
    counts=summary['splits']
    return f'''# 迈克尔逊光谱C方案：按采集位置划分的数据包

4000条光谱按顺序形成400个候选采集位置，每位置10条。顺序假设由OPD阶跃与FFT幅度形状强力支持，但没有硬件位置日志独立证明。

## 数据在哪里

| 分区 | 位置数 | 光谱数 | 文件 |
|---|---:|---:|---|
| 训练 | {counts['train']['positions']} | {counts['train']['samples']} | datasets/train/processed_C_with_pseudo_labels.npz |
| 验证 | {counts['validation']['positions']} | {counts['validation']['samples']} | datasets/validation/processed_C_with_pseudo_labels.npz |
| 测试 | {counts['test']['positions']} | {counts['test']['samples']} | datasets/test/processed_C_with_pseudo_labels.npz |

每个分区还包含原始CSV的raw目录及samples_and_labels.csv。全部4000条保留。代码位于code，中文原理说明位于docs。

## 使用

```python
import numpy as np
train = np.load('datasets/train/processed_C_with_pseudo_labels.npz')
X = train['X']                       # float32，样本数×2048
y_opd = train['opd_pseudo_m']         # OPD伪标签，单位m
y_sin = train['phase_sin']           # 1560nm参考点的中心包裹相位
y_cos = train['phase_cos']
opd_mask = train['opd_valid']
phase_mask = train['phase_valid']
position_review = train['position_review']
```

OPD与相位是算法伪标签，不是独立标定真值。损失计算前先按各自mask选有效目标；position_review表示额外待核对记录，不自动删除。

## 分组与泄漏检查

随机种子{summary['seed']}；按位置70/15/15划分。3处跨位置完全重复谱涉及的位置绑定到同一分区。同位置及完全重复谱均没有跨分区。

grouping/sample_manifest.csv逐条列出位置、重复编号、分区和质量标志；position_manifest.csv逐位置核查；position_needs_review.csv保留5条位置一致性待核对记录，pseudo_label_needs_review.csv保留8条标签QC复核记录。两类标志有重叠，不是13条不同样本。

这是随机留出未见位置的同次采集评估，以插值为主，可能包含训练OPD范围外的边界位置；split_summary.json分别记录数量。它不是专门设置的连续区间外推、跨日期或跨设备验证。本次曾用全数据探索预处理，正式论文的独立精度结论仍需要独立参考/补采数据。

## 代码复现

进入code目录，安装requirements.txt。对某分区重新做C预处理：

```powershell
python -m pip install -r requirements.txt
python -X utf8 scripts/preprocess_c.py --data-dir ../datasets/train/raw --out ../reproduced/train_C
python -X utf8 scripts/generate_pseudo_labels.py --data-dir ../datasets/train/raw --inputs ../reproduced/train_C/model_inputs_C_2048.npz --out ../reproduced/train_labels
python -m pytest tests -q
```

若要完整重建本划分，需要将三个分区的raw文件复制汇总到一个新的空目录，再使用source_artifacts/full_C_with_pseudo_labels.npz作为划分脚本的--labeled-data，指定汇总目录为--raw-dir，并使用新的--out目录。不要移动或覆盖本包原文件。
'''


# ==================== 第6块：运行入口 ====================
def main():
    parser=argparse.ArgumentParser(description='每位置10条核查、无组泄漏划分和桌面数据包')
    parser.add_argument('--labeled-data',type=Path,default=ROOT/'analysis/pseudo_labels/training_C_with_pseudo_labels.npz')
    parser.add_argument('--raw-dir',type=Path,default=ROOT/'analysis/data_4000')
    parser.add_argument('--out',type=Path,default=Path.home()/'Desktop/迈克尔逊光谱_C方案_分组数据集_20261117')
    parser.add_argument('--seed',type=int,default=20261117)
    parser.add_argument('--repeats',type=int,default=10)
    parser.add_argument('--no-raw',action='store_true',help='仅导出处理后数组，不复制原始CSV')
    args=parser.parse_args()
    summary=build_package(args.labeled_data,args.raw_dir,args.out,args.repeats,args.seed,not args.no_raw)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
