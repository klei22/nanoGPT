import argparse
import wave
import numpy as np
import csv
from tqdm import tqdm


def load_wav(path):
    with wave.open(path, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        audio = wf.readframes(n_frames)
        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        sampwidth = wf.getsampwidth()
        dtype = dtype_map.get(sampwidth)
        if dtype is None:
            raise ValueError(f"Unsupported sample width: {sampwidth}")
        data = np.frombuffer(audio, dtype=dtype).astype(np.float32)
        if wf.getnchannels() == 2:
            data = data.reshape(-1, 2).mean(axis=1)
        data /= np.iinfo(dtype).max
    return data, sample_rate


def stft(audio, n_fft=400, hop_length=160):
    window = np.hanning(n_fft)
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    spectrogram = np.empty((n_frames, n_fft // 2 + 1), dtype=np.float32)
    for i in tqdm(range(n_frames), desc='Computing FFT'):
        start = i * hop_length
        frame = audio[start:start + n_fft] * window
        spec = np.fft.rfft(frame, n=n_fft)
        spectrogram[i] = np.abs(spec)
    return spectrogram


def save_csv(spectrogram, path):
    header = [f'bin_{i}' for i in range(spectrogram.shape[1])]
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in spectrogram:
            writer.writerow(row.tolist())


def main():
    parser = argparse.ArgumentParser(
        description='Compute FFT per time step similar to Whisper and save as CSV.'
    )
    parser.add_argument('input_wav', help='Path to WAV file (16kHz mono recommended)')
    parser.add_argument('output_csv', help='Output CSV file path')
    parser.add_argument('--n_fft', type=int, default=400, help='FFT window size')
    parser.add_argument('--hop_length', type=int, default=160, help='Hop length')
    args = parser.parse_args()

    audio, sr = load_wav(args.input_wav)
    if sr != 16000:
        print(f'Warning: expected 16000 Hz audio for Whisper, got {sr} Hz.')
    spec = stft(audio, args.n_fft, args.hop_length)
    save_csv(spec, args.output_csv)


if __name__ == '__main__':
    main()
