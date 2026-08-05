#!/usr/bin/env python3
"""Utilities for mel-spectrogram integer multicontext datasets and demos."""
from __future__ import annotations

import argparse, csv, json, pickle, re, shutil, sys, wave
from array import array
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MEL_DIR = REPO_ROOT / "data" / "mel_spectrogram"
sys.path.insert(0, str(MEL_DIR))
from audio_to_token_mel import CSV_METADATA_PREFIX, compact_json  # noqa: E402
from token_mel_to_audio import canonical_state_crc, read_csv_container  # noqa: E402


def mel_columns(header: Sequence[str]) -> list[tuple[int, str]]:
    cols = [(i, h.strip()) for i, h in enumerate(header) if re.fullmatch(r"mel_\d+_q", h.strip())]
    if not cols:
        raise ValueError("No mel_NNN_q columns found.")
    return cols


def dtype_for_vocab(vocab: int):
    return ("I", "uint32") if vocab > 65536 else ("H", "uint16")


def read_mel_csv(path: Path):
    metadata = None
    rows = []
    with path.open(newline='', encoding='utf-8') as f:
        header = None
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(CSV_METADATA_PREFIX):
                metadata = json.loads(s[len(CSV_METADATA_PREFIX):])
                continue
            if s.startswith('#'):
                continue
            header = next(csv.reader([line]))
            break
        if header is None:
            raise ValueError("CSV has no header")
        cols = mel_columns(header)
        for row in csv.reader(f):
            if not row or (row[0].startswith('#')):
                continue
            rows.append([int(row[i]) for i, _ in cols])
    if len(rows) < 2:
        raise ValueError("Need at least two mel rows")
    return header, cols, np.asarray(rows, dtype=np.int64), metadata



def metadata_signature(metadata: dict | None) -> tuple:
    if not metadata:
        return ()
    shape = metadata.get("shape", {})
    waveform = metadata.get("waveform", {})
    stft = metadata.get("stft", {})
    mel = metadata.get("mel", {})
    quantizer = metadata.get("quantizer", {})
    return (
        int(shape.get("bands", -1)),
        int(waveform.get("sample_rate", -1)),
        int(stft.get("n_fft", -1)),
        int(stft.get("win_length", -1)),
        int(stft.get("hop_length", -1)),
        int(mel.get("n_mels", -1)),
        float(mel.get("fmin", -1.0)),
        float(mel.get("fmax", -1.0)),
        int(quantizer.get("levels", -1)),
        float(quantizer.get("db_min", 0.0)),
        float(quantizer.get("db_max", 0.0)),
        metadata.get("profile"),
    )


def metadata_for_states(reference: dict | None, states: np.ndarray) -> dict | None:
    if not reference:
        return None
    metadata = dict(reference)
    metadata["shape"] = dict(metadata["shape"])
    metadata["shape"]["timesteps"] = int(states.shape[0])
    metadata["waveform"] = dict(metadata["waveform"])
    metadata["waveform"]["decoded_sample_count"] = int(
        states.shape[0] * int(metadata["stft"]["hop_length"])
    )
    metadata["csv"] = dict(metadata.get("csv", {}))
    metadata["csv"]["time_columns_included"] = False
    metadata["integrity"] = dict(metadata["integrity"])
    metadata["integrity"]["state_crc32"] = (
        f"{canonical_state_crc(states, int(metadata['quantizer']['levels'])):08x}"
    )
    return metadata


def write_mel_state_csv(path: Path, columns: Sequence[str], states: np.ndarray, metadata: dict | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        if metadata:
            f.write(CSV_METADATA_PREFIX + compact_json(metadata) + "\n")
        writer.writerows(states.astype(int).tolist())

def write_bin(path: Path, values: Sequence[int], typecode: str) -> None:
    with path.open('wb') as f:
        array(typecode, values).tofile(f)



def cmd_concat_csv(args):
    inputs = [Path(value).resolve() for value in args.input_csvs]
    if not inputs:
        raise ValueError("At least one input CSV is required.")
    first_header, first_cols, first_states, first_metadata = read_mel_csv(inputs[0])
    column_names = [name for _, name in first_cols]
    signature = metadata_signature(first_metadata)
    all_states = [first_states]
    manifest_inputs = []
    for path, states, metadata in [(inputs[0], first_states, first_metadata)]:
        manifest_inputs.append({"path": str(path), "frames": int(states.shape[0])})
    for input_csv in inputs[1:]:
        _, cols, states, metadata = read_mel_csv(input_csv)
        names = [name for _, name in cols]
        if names != column_names:
            raise ValueError(f"Mel columns in {input_csv} do not match {inputs[0]}.")
        if signature and metadata_signature(metadata) != signature:
            raise ValueError(f"Mel metadata settings in {input_csv} do not match {inputs[0]}.")
        all_states.append(states)
        manifest_inputs.append({"path": str(input_csv), "frames": int(states.shape[0])})
    concatenated = np.concatenate(all_states, axis=0)
    output_metadata = metadata_for_states(first_metadata, concatenated)
    output_csv = Path(args.output_csv).resolve()
    write_mel_state_csv(output_csv, column_names, concatenated, output_metadata)
    if args.manifest_json:
        with Path(args.manifest_json).open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "output_csv": str(output_csv),
                    "total_frames": int(concatenated.shape[0]),
                    "mel_columns": len(column_names),
                    "inputs": manifest_inputs,
                },
                f,
                indent=2,
            )
    print(output_csv)

def cmd_prepare(args):
    input_csv = Path(args.input_csv).resolve()
    _, cols, states, metadata = read_mel_csv(input_csv)
    levels = args.states_per_column or int((metadata or {}).get('quantizer', {}).get('levels', 64))
    if states.min() < 0 or states.max() >= levels:
        raise ValueError(f"Mel states [{states.min()}, {states.max()}] exceed [0,{levels-1}]")
    out = REPO_ROOT / 'data' / args.output_root
    out.mkdir(parents=True, exist_ok=True)
    train_n = int(states.shape[0] * args.train_ratio)
    typecode, dtype_name = dtype_for_vocab(levels)
    manifest = {"tokenizer":"mel_integer_range_multicontext_manifest","source_mel_csv":str(input_csv),"rows":int(states.shape[0]),"train_rows":train_n,"val_rows":int(states.shape[0]-train_n),"output_root":args.output_root,"multicontext_datasets":[],"columns":[],"roundtrip_metadata":metadata}
    for j, (_, name) in enumerate(cols):
        cdir = out / name
        cdir.mkdir(parents=True, exist_ok=True)
        vals = states[:, j].astype(int).tolist()
        write_bin(cdir/'train.bin', vals[:train_n], typecode)
        write_bin(cdir/'val.bin', vals[train_n:], typecode)
        meta = {"tokenizer":"csv_integer_range","vocab_size":levels,"source_csv":str(input_csv),"source_column":name,"context_name":name,"column_index":j,"has_header":True,"int_min":0,"int_max":levels-1,"value_encoding":"token_id = raw_integer_value","value_decoding":"raw_integer_value = token_id","dtype":dtype_name,"samples":int(states.shape[0]),"train_ratio":float(args.train_ratio),"mel_mc_int":True}
        with (cdir/'meta.pkl').open('wb') as f: pickle.dump(meta, f)
        manifest['multicontext_datasets'].append(f"{args.output_root}/{name}")
        manifest['columns'].append(meta)
    with (out/'manifest.json').open('w', encoding='utf-8') as f: json.dump(manifest, f, indent=2)
    print(out)


def cmd_cut_prompt(args):
    src = Path(args.mel_csv).resolve()
    header, cols, states, metadata = read_mel_csv(src)
    hop = int(metadata['stft']['hop_length']) if metadata else None
    sr = int(metadata['waveform']['sample_rate']) if metadata else None
    if hop and sr:
        n = max(1, min(states.shape[0], int(np.ceil(args.cutoff_s * sr / hop))))
    else:
        n = max(1, min(states.shape[0], int(round(args.cutoff_s / args.timestep_ms * 1000.0))))
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'prompt.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow([name for _, name in cols]); w.writerows(states[:n].tolist())
    bin_dir = out_dir / 'bins'; bin_dir.mkdir(exist_ok=True)
    levels = int((metadata or {}).get('quantizer', {}).get('levels', 64)); typecode, dtype_name = dtype_for_vocab(levels)
    for j, (_, name) in enumerate(cols): write_bin(bin_dir / f'{name}.bin', states[:n, j].astype(int).tolist(), typecode)
    with (out_dir/'prompt_manifest.json').open('w') as f: json.dump({"csv":str(csv_path),"bin_dir":str(bin_dir),"start_files":[str(bin_dir/f'{name}.bin') for _, name in cols],"dtype":dtype_name,"cutoff_s":args.cutoff_s,"prompt_frames":n}, f, indent=2)
    print(csv_path)


def cmd_wrap_csv(args):
    input_csv = Path(args.input_csv); ref = read_csv_container(Path(args.reference_mel_csv))
    with input_csv.open(newline='', encoding='utf-8') as f:
        reader=csv.reader(f); header=next(reader); cols=mel_columns(header); states=np.asarray([[int(r[i]) for i,_ in cols] for r in reader if r], dtype=np.int64)
    md = metadata_for_states(ref.metadata, states)
    out=Path(args.output_csv); out.parent.mkdir(parents=True, exist_ok=True)
    with input_csv.open('r', encoding='utf-8') as src, out.open('w', encoding='utf-8') as dst:
        dst.write(src.readline()); dst.write(CSV_METADATA_PREFIX + compact_json(md) + '\n'); shutil.copyfileobj(src, dst)
    print(out)


def cmd_stitch(args):
    def read_wav(p):
        with wave.open(str(p),'rb') as w:
            params=w.getparams(); data=w.readframes(w.getnframes())
        return params, data
    p1,d1=read_wav(Path(args.prompt_wav)); p2,d2=read_wav(Path(args.continuation_wav))
    if p1[:3] != p2[:3] or p1.framerate != p2.framerate: raise ValueError('WAV formats differ')
    out=Path(args.output_wav); out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out),'wb') as w: w.setparams(p1); w.writeframes(d1+d2)
    print(out)


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(required=True)
    a=sub.add_parser('concat-csv'); a.add_argument('input_csvs', nargs='+'); a.add_argument('--output_csv', required=True); a.add_argument('--manifest_json'); a.set_defaults(func=cmd_concat_csv)
    a=sub.add_parser('prepare'); a.add_argument('input_csv'); a.add_argument('--output_root',default='mel_mc_int'); a.add_argument('--train_ratio',type=float,default=.9); a.add_argument('--states_per_column',type=int); a.set_defaults(func=cmd_prepare)
    a=sub.add_parser('cut-prompt'); a.add_argument('mel_csv'); a.add_argument('--cutoff_s',type=float,required=True); a.add_argument('--output_dir',default='mel_mc_int_prompt'); a.add_argument('--timestep_ms',type=float,default=15); a.set_defaults(func=cmd_cut_prompt)
    a=sub.add_parser('wrap-csv'); a.add_argument('input_csv'); a.add_argument('--reference_mel_csv',required=True); a.add_argument('--output_csv',required=True); a.set_defaults(func=cmd_wrap_csv)
    a=sub.add_parser('stitch-wav'); a.add_argument('--prompt_wav',required=True); a.add_argument('--continuation_wav',required=True); a.add_argument('--output_wav',required=True); a.set_defaults(func=cmd_stitch)
    args=p.parse_args(); args.func(args)
if __name__=='__main__': main()
