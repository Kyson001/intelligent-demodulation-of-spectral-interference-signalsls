"""按光谱峰间距法生成OPD伪标签，并生成固定参考点的相位伪标签。

运行：python -X utf8 scripts/generate_pseudo_labels.py
依赖现有C方案脚本和src/demodulation/pseudo_labels.py，所有步骤均有中文说明。
"""
# ==================== 第1块：导入和项目路径 ====================
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from scripts.preprocess_c import CConfig,read_spectrum,preprocess_spectrum
from demodulation.pseudo_labels import LabelConfig,label_prepared


def join_with_c_inputs(input_path, labels, rows, output_path, cconfig):
    """按ID、原始文件哈希及预处理配置三重检查后，将伪标签与C输入合并。

    防止同名文件被替换后，误把新标签拼到旧输入；不满足检查时只保存标签，不强行合并。
    """
    if not input_path.exists():
        return False,'C_input_file_missing'
    quality_path=input_path.parent/'quality_and_position_review.csv'
    manifest_path=input_path.parent/'manifest_C.json'
    if not quality_path.exists() or not manifest_path.exists():
        return False,'C_input_provenance_missing'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['config']!=cconfig.__dict__:
        return False,'C_preprocessing_config_mismatch'
    with quality_path.open(encoding='utf-8-sig',newline='') as stream:
        prior_hashes={r['sample_id']:r['source_sha256'] for r in csv.DictReader(stream)}
    inputs=np.load(input_path)
    ids=labels['sample_ids']
    mapping={name:i for i,name in enumerate(ids)}
    input_ids=inputs['sample_ids']
    if len(mapping)!=len(input_ids) or set(mapping)!=set(input_ids):
        return False,'sample_ids_mismatch'
    if any(prior_hashes.get(r['sample_id'])!=r['source_sha256'] for r in rows):
        return False,'raw_source_hash_mismatch'
    order=np.array([mapping[name] for name in input_ids])
    bundled={key:inputs[key] for key in inputs.files}
    bundled.update({key:value[order] if value.ndim==1 and len(value)==len(ids) else value
                    for key,value in labels.items()})
    hashes={r['sample_id']:r['source_sha256'] for r in rows}
    bundled['sample_ids']=input_ids
    bundled['source_sha256']=np.array([hashes[name] for name in input_ids])
    bundled['split']=np.full(len(input_ids),'unassigned')
    np.savez_compressed(output_path,**bundled)
    return True,'sample_id_raw_sha256_and_C_config_verified'


def main():
    # ==================== 第2块：命令行参数与输出保护 ====================
    parser=argparse.ArgumentParser(description='多干涉峰频率间距OPD＋中心相位伪标签')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'analysis/data_4000')
    parser.add_argument('--out',type=Path,default=ROOT/'analysis/pseudo_labels')
    parser.add_argument('--inputs',type=Path,default=ROOT/'analysis/preprocessed_C/model_inputs_C_2048.npz',
                        help='已有C输入矩阵，用于核对sample_id；不从2048点输入反推标签')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/pseudo_labels.json')
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--limit',type=int,default=0,help='只用于调试；正式运行默认全部')
    args=parser.parse_args()
    config=LabelConfig(**json.loads(args.config.read_text(encoding='utf-8')))
    training=json.loads((ROOT/'configs/training_preprocessing.json').read_text(encoding='utf-8'))
    unit=training.get('intensity_unit','recorded')
    cconfig=CConfig(background_sigma_points=training['background_sigma_points'],
                    envelope_floor_fraction=training['envelope_floor_fraction'],
                    model_points=training['model_points'],
                    intensity_unit='dbm' if unit=='dbm' else 'recorded')
    paths=sorted(args.data_dir.glob('file*.csv'))
    if args.limit:
        paths=paths[:args.limit]
    if not paths:
        raise ValueError('没有找到file*.csv')
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'pseudo_labels.csv').exists() and not args.overwrite:
        raise FileExistsError('标签已经存在；需要重算时指定新的--out或添加--overwrite')
    trace_dir=args.out/'traces'
    trace_dir.mkdir(exist_ok=True)
    rows=[]

    # ==================== 第3块：逐条从原始高分辨率光谱制作标签 ====================
    # 不平均每位置10条，不按10 μm编号设标签，不读取旧距离缓存作为真值。
    for index,path in enumerate(paths):
        row=dict(sample_id=path.stem,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 label_source='spectral_peak_spacing_and_local_fourier_phase',
                 reference_wavelength_nm=config.reference_nm,
                 opd_peak_m=np.nan,half_opd_m=np.nan,phase_pseudo_rad=np.nan,
                 phase_sin=np.nan,phase_cos=np.nan,opd_valid=False,phase_valid=False,
                 status='failed',quality_reasons='')
        try:
            wavelength,intensity=read_spectrum(path)
            stages=preprocess_spectrum(wavelength,intensity,cconfig)
            result,trace=label_prepared(stages,config)
            row.update(result)
            # 保存固定的首/中/末样本峰位，便于复核实际寻到了哪些峰。
            if index in {0,(len(paths)-1)//2,len(paths)-1}:
                np.savez_compressed(trace_dir/f'{path.stem}_label_trace.npz',
                                    k_dense=stages['k_dense'],
                                    corrected_before_hann=stages['envelope_corrected'],
                                    windowed_c=stages['windowed_c'],**trace)
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
            # 失败样本仍保留一行，用valid掩码控制训练；绝不静默删除或填假标签。
            row['quality_reasons']=str(error)
        rows.append(row)
        if (index+1)%500==0:
            print(f'已制作伪标签 {index+1}/{len(paths)} 条',flush=True)

    # ==================== 第4块：保存可读CSV与机器学习标签数组 ====================
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with (args.out/'pseudo_labels.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (args.out/'needs_review.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(r for r in rows if r['status']!='accepted')
    ids=np.array([r['sample_id'] for r in rows])
    labels=dict(sample_ids=ids,
                opd_pseudo_m=np.array([r['opd_peak_m'] for r in rows]),
                phase_pseudo_rad=np.array([r['phase_pseudo_rad'] for r in rows]),
                phase_sin=np.array([r['phase_sin'] for r in rows]),
                phase_cos=np.array([r['phase_cos'] for r in rows]),
                opd_valid=np.array([r['opd_valid'] for r in rows],dtype=bool),
                phase_valid=np.array([r['phase_valid'] for r in rows],dtype=bool),
                reference_wavelength_nm=np.array(config.reference_nm),
                label_status=np.array('algorithmic_pseudo_labels_not_independent_truth'))
    np.savez_compressed(args.out/'pseudo_labels.npz',**labels)

    # ==================== 第5块：与已有C输入对齐，不自动划分数据集 ====================
    # 只按明确sample_id对齐，不按CSV行号盲目拼接。保留待确认的采集位置和分区。
    joined,join_provenance=join_with_c_inputs(args.inputs,labels,rows,
                                            args.out/'training_C_with_pseudo_labels.npz',cconfig)

    # ==================== 第6块：汇总QC与可追溯设置 ====================
    valid_opd=labels['opd_valid']; valid_phase=labels['phase_valid']
    reason_counts={}
    for r in rows:
        for reason in r['quality_reasons'].split(';'):
            if reason:
                reason_counts[reason]=reason_counts.get(reason,0)+1
    summary=dict(samples=len(rows),opd_valid=int(valid_opd.sum()),phase_valid=int(valid_phase.sum()),
                 status_counts={s:sum(r['status']==s for r in rows) for s in ['accepted','review','failed']},
                 opd_range_mm=[float(np.nanmin(labels['opd_pseudo_m'])*1e3),float(np.nanmax(labels['opd_pseudo_m'])*1e3)]
                              if np.isfinite(labels['opd_pseudo_m']).any() else None,
                 peak_count_median=float(np.median([r['peak_count'] for r in rows if 'peak_count' in r]))
                                   if any('peak_count' in r for r in rows) else None,
                 quality_reason_counts=reason_counts,config=config.to_dict(),preprocessing_config=cconfig.__dict__,
                 joined_to_C_inputs=joined,join_provenance=join_provenance,
                 split_status='unassigned_pending_confirmed_positions',
                 phase_definition='wrapped phase at reference wavelength; positive OPD branch; unknown 2pi order',
                 opd_definition='effective group OPD averaged over accepted peak band; not geometric mirror displacement',
                 truth_status='pseudo labels; no independent accuracy established',
                 label_csv_sha256=hashlib.sha256((args.out/'pseudo_labels.csv').read_bytes()).hexdigest())
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
