"""Export all 4000 training-ready inputs; legacy labels remain explicitly unverified."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from demodulation.preprocessing import Config, model_input


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=ROOT/'analysis/data_4000')
    parser.add_argument('--out',type=Path,default=ROOT/'analysis/preprocessed')
    parser.add_argument('--scheme',choices=['A','B'],default='A')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    config=Config(**json.loads((ROOT/'configs/preprocessing_comparison.json').read_text(encoding='utf-8')))
    paths=sorted(args.data_dir.glob('file*.csv'))
    if not paths:
        raise ValueError('No input CSVs')
    signals,ids,groups,scales,means=[],[],[],[],[]
    common_k=None
    for index,path in enumerate(paths):
        a=np.loadtxt(path,delimiter=',',skiprows=1)
        x,k,meta=model_input(a[:,0],a[:,1],args.scheme,config)
        if common_k is None:
            common_k=k
        np.testing.assert_allclose(k,common_k,rtol=0,atol=1e-8)
        signals.append(x); ids.append(path.stem)
        groups.append(hashlib.sha256(a[:,1].tobytes()).hexdigest())
        scales.append(meta['rms']); means.append(meta['mean'])
        if (index+1)%500==0:
            print(f'Exported {index+1}/{len(paths)} inputs',flush=True)
    unique=np.array(sorted(set(groups)))
    rng=np.random.default_rng(20261116)
    rng.shuffle(unique)
    a=int(.7*len(unique)); b=int(.85*len(unique))
    mapping={g:'train' if i<a else 'validation' if i<b else 'test' for i,g in enumerate(unique)}
    splits=np.array([mapping[g] for g in groups])
    label_paths=list((args.data_dir/'_cache').glob('d_mm*.npy'))
    legacy=np.load(label_paths[0]) if label_paths else np.full(len(paths),np.nan)
    # NEVER output legacy distance under the name opd_truth.
    X=np.stack(signals)
    assert np.isfinite(X).all()
    for left,right in [('train','validation'),('train','test'),('validation','test')]:
        assert not (set(np.array(groups)[splits==left])&set(np.array(groups)[splits==right]))
    filename=args.out/f'model_inputs_{args.scheme}_{config.model_points}.npz'
    np.savez_compressed(filename,X=X,k_rad_per_m=common_k,sample_ids=np.array(ids),
                        duplicate_group=np.array(groups),split=splits,rms=np.array(scales),mean=np.array(means),
                        legacy_d_mm=legacy,label_status=np.array('unverified_legacy_distance_no_phase_truth'))
    summary=dict(scheme=args.scheme,shape=list(X.shape),dtype=str(X.dtype),
                 unique_groups=len(unique),counts={s:int(sum(splits==s)) for s in ['train','validation','test']},
                 split_seed=20261116,duplicate_group_leakage=False,
                 split_limit='Only exact duplicates identified; same-batch correlation still unknown',
                 evaluation_limit='Exploratory preprocessing used full dataset; future evaluation needs independently frozen data',
                 label_status='Original workbook and independent phase labels missing',
                 file_sha256=hashlib.sha256(filename.read_bytes()).hexdigest())
    (args.out/f'model_inputs_{args.scheme}_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
