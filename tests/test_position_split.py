"""按采集位置划分：检查顺序证据、重复绑定与确定性，避免样本级随机泄漏。"""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.split_dataset_by_position import audit_ordering,position_components,assign_positions


def test_exact_position_counts_duplicates_and_determinism():
    positions=np.repeat(np.arange(1,401),10)
    duplicate=np.array([f's{i:04}' for i in range(4000)])
    duplicate[89]=duplicate[90]
    duplicate[1449]=duplicate[1450]
    duplicate[3059]=duplicate[3060]
    components,cross=position_components(positions,duplicate)
    assert len(components)==397 and len(cross)==3
    a,groups=assign_positions(components,400)
    b,_=assign_positions(components,400)
    assert a==b
    assert [sum(v==s for v in a.values()) for s in ['train','validation','test']]==[280,60,60]
    for x,y in [(9,10),(145,146),(306,307)]:
        assert a[x]==a[y] and groups[x]==groups[y]
    for split_a,split_b in [('train','validation'),('train','test'),('validation','test')]:
        indices_a=[i for i,p in enumerate(positions) if a[int(p)]==split_a]
        indices_b=[i for i,p in enumerate(positions) if a[int(p)]==split_b]
        assert not set(duplicate[indices_a])&set(duplicate[indices_b])


def test_order_gate_accepts_stepwise_blocks_rejects_wrong_block_origin():
    rng=np.random.default_rng(4)
    opd=np.repeat(np.arange(20)*20e-6+2e-3,10)+rng.normal(0,.1e-6,200)
    # 幅度形状输入只用作第二诊断；核心门槛来自OPD边界和错位对照。
    data=dict(sample_ids=np.array([f'file{i:04}' for i in range(1,201)]),
              opd_pseudo_m=opd,X=rng.normal(size=(200,2048)).astype(np.float32))
    report,offsets,_,_=audit_ordering(data)
    assert report['best_offset']==0 and report['positions']==20
    assert offsets[0]['separation_ratio']>5
    data['opd_pseudo_m']=np.roll(opd,3)
    with pytest.raises(ValueError,match='没有充分支持'):
        audit_ordering(data)
    data['sample_ids'][0]='file0002'
    with pytest.raises(ValueError,match='编号'):
        audit_ordering(data)
