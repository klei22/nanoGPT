#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct audio from Whisper-style mel CSV.")
    parser.add_argument("csv_path", type=Path, help="Path to the mel CSV file.")
    parser.add_argument("--output", type=Path, default=Path("reconstructed.wav"), help="Output WAV path.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate.")
    parser.add_argument("--n_fft", type=int, default=400, help="FFT size.")
    parser.add_argument("--hop_length", type=int, default=160, help="Hop length.")
    parser.add_argument("--win_length", type=int, default=400, help="Window length.")
    parser.add_argument("--n_mels", type=int, default=80, help="Number of mel channels.")
    parser.add_argument("--f_min", type=float, default=0.0, help="Minimum frequency.")
    parser.add_argument("--f_max", type=float, default=8000.0, help="Maximum frequency.")
    parser.add_argument("--center", action=argparse.BooleanOptionalAction, default=True, help="Center frames.")
    parser.add_argument("--power", type=float, default=2.0, help="Spectrogram power.")
    parser.add_argument("--normalized_input", action=argparse.BooleanOptionalAction, default=True,
                        help="CSV is Whisper-normalized log-mel (default).")
    parser.add_argument("--griffin_lim_iters", type=int, default=32,
                        help="Number of Griffin-Lim iterations.")
    parser.add_argument("--mel_scale", type=str, default="slaney", help="Mel scale to use.")
    parser.add_argument("--mel_norm", type=str, default="slaney", help="Mel filterbank normalization.")
    return parser.parse_args()


def inverse_whisper_normalize(mel):
    return torch.pow(10.0, mel * 4.0 - 4.0)


def main():
    args = parse_args()
    mel = np.loadtxt(args.csv_path, delimiter=",", dtype=np.float32)
    if mel.ndim == 1:
        mel = mel[np.newaxis, :]
    mel_tensor = torch.from_numpy(mel).transpose(0, 1)

    if args.normalized_input:
        mel_tensor = inverse_whisper_normalize(mel_tensor)

    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=args.n_fft // 2 + 1,
        n_mels=args.n_mels,
        sample_rate=args.sample_rate,
        f_min=args.f_min,
        f_max=args.f_max,
        mel_scale=args.mel_scale,
        norm=args.mel_norm,
    )
    linear_spec = inverse_mel(mel_tensor)

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=args.n_fft,
        n_iter=args.griffin_lim_iters,
        hop_length=args.hop_length,
        win_length=args.win_length,
        power=args.power,
        center=args.center,
    )
    waveform = griffin_lim(linear_spec)
    waveform = waveform.unsqueeze(0)
    torchaudio.save(str(args.output), waveform, args.sample_rate)


if __name__ == "__main__":
    main()
