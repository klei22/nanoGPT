#!/usr/bin/env python3
"""Decompose audio into quantized Whisper-style mel CSV for multicontext training."""
from __future__ import annotations
import argparse, csv, json, pickle, subprocess, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('audio', help='Input wav/mp3/flac audio')
    p.add_argument('--output_root', default='audio_mel_csv', help='Folder under data/ for datasets and manifest')
    p.add_argument('--work_dir', default='data/audio_mel_csv/work')
    p.add_argument('--quantized_csv', default=None)
    p.add_argument('--scale', type=int, default=32767, help='Scale normalized mel floats to integer range')
    p.add_argument('--train_ratio', type=float, default=0.9)
    p.add_argument('--keep_float_csv', action='store_true')
    args=p.parse_args()
    import numpy as np
    repo=Path(__file__).resolve().parents[2]
    work=Path(args.work_dir); work = work if work.is_absolute() else repo/work; work.mkdir(parents=True, exist_ok=True)
    float_csv=work/'mel_float.csv'
    q_csv=Path(args.quantized_csv) if args.quantized_csv else work/'mel_int.csv'
    if not q_csv.is_absolute(): q_csv=repo/q_csv
    prep=repo/'data/template/prepare.py'
    subprocess.run([sys.executable, str(prep), '--method','whisper_mel_csv','--train_input',args.audio,'--train_output',str(float_csv),'--percentage_train','1.0'], check=True, cwd=repo)
    mel=np.loadtxt(float_csv, delimiter=',', dtype=np.float32)
    if mel.ndim==1: mel=mel[np.newaxis,:]
    q=np.rint(np.clip(mel,0.0,1.0)*args.scale).astype(np.int32)
    q_csv.parent.mkdir(parents=True, exist_ok=True)
    header=[f'mel_{i:03d}' for i in range(q.shape[1])]
    with q_csv.open('w', newline='', encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(header); w.writerows(q.tolist())
    out_root=args.output_root
    subprocess.run([sys.executable, str(repo/'data/csv_mc_int/prepare_csv_integer_multicontext.py'), '--input_csv', str(q_csv), '--output_root', out_root, '--train_ratio', str(args.train_ratio), '--default_range', f'0:{args.scale}', '--allow_out_of_range'], check=True, cwd=repo)
    manifest=repo/'data'/out_root/'manifest.json'
    meta={}
    with (repo/'meta.pkl').open('rb') as f: meta=pickle.load(f)
    m=json.loads(manifest.read_text())
    m.update({'audio_source': str(Path(args.audio).resolve()), 'float_mel_csv': str(float_csv), 'quantized_mel_csv': str(q_csv), 'quantization': {'scale': args.scale, 'inverse': 'float = integer / scale'}, 'mel_meta': meta})
    manifest.write_text(json.dumps(m, indent=2), encoding='utf-8')
    if not args.keep_float_csv and float_csv.exists():
        pass
    print(f'Quantized mel CSV: {q_csv}')
    print(f'Manifest: {manifest}')
if __name__=='__main__': main()
