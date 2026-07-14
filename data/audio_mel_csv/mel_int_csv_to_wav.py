#!/usr/bin/env python3
"""Convert quantized mel integer CSV back to WAV via existing mel_csv_to_wav.py."""
from __future__ import annotations
import argparse, csv, subprocess, sys, tempfile
from pathlib import Path

def load_numeric(path):
    import numpy as np
    with Path(path).open(newline='', encoding='utf-8') as f:
        rows=list(csv.reader(f))
    if not rows: raise ValueError('empty CSV')
    start=0
    try: [float(x) for x in rows[0]]
    except ValueError: start=1
    return np.array([[float(x) for x in r] for r in rows[start:] if r], dtype=np.float32)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv_path'); p.add_argument('--output', default='reconstructed.wav'); p.add_argument('--scale', type=float, default=32767.0)
    p.add_argument('--keep_float_csv', default=None)
    p.add_argument('--griffin_lim_iters', type=int, default=32)
    args=p.parse_args(); repo=Path(__file__).resolve().parents[2]
    import numpy as np
    data=load_numeric(args.csv_path)/args.scale
    tmp_path=Path(args.keep_float_csv) if args.keep_float_csv else Path(tempfile.mkstemp(suffix='.csv')[1])
    np.savetxt(tmp_path, data, delimiter=',', fmt='%.6f')
    try:
        subprocess.run([sys.executable, str(repo/'data/template/mel_csv_to_wav.py'), str(tmp_path), '--output', args.output, '--griffin_lim_iters', str(args.griffin_lim_iters)], check=True, cwd=repo)
    finally:
        if not args.keep_float_csv: tmp_path.unlink(missing_ok=True)
if __name__=='__main__': main()
