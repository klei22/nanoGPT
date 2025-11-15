# prepare.py
import json
import os
import argparse
import numpy as np
from tokenizers import (
    SentencePieceTokenizer,
    TiktokenTokenizer,
    CustomTokenizer,
    ByteTokenizer,
    CharTokenizer,
    CustomCharTokenizerWithByteFallback,
    JsonByteTokenizerWithByteFallback,
    SineWaveTokenizer,
    AudioFFTTokenizer,
)
from tqdm import tqdm
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser(description="Tokenize text data using different methods.")

    # Input/output arguments
    parser.add_argument("-t", "--train_input", type=str, required=True, help="Path to the input text file")
    parser.add_argument("-v", "--val_input", type=str, help="Path to validation input file. If not provided, train_input will be split using percentage_train")
    parser.add_argument("--train_output", type=str, default="train.bin", help="Path to save the training output file")
    parser.add_argument("--val_output", type=str, default="val.bin", help="Path to save the validation output file")
    parser.add_argument("-p", "--percentage_train", type=float, default=0.9, help="Percentage of data to use for training (between 0 and 1) when val_input is not provided")

    # Tokenizer selection and configuration
    parser.add_argument("--method", type=str,
                       choices=["sentencepiece", "tiktoken", "char", "custom", "byte", "custom_char_byte_fallback", "json_byte_fallback", "sinewave", "audio_fft"],
                       default="tiktoken", help="Tokenization method")

    # Sine wave tokenizer arguments
    parser.add_argument("--sine_period", type=float, default=1.0,
                        help="Period multiplier applied to the sine wave (in radians)")
    parser.add_argument("--sine_points_per_period", type=int, default=64,
                        help="Number of discrete points sampled per sine wave period")
    parser.add_argument("--sine_num_periods", type=int, default=10,
                        help="Total number of periods to generate")
    parser.add_argument("--sine_amplitude", type=float, default=50.0,
                        help="Amplitude of the generated sine wave prior to clamping")

    # SentencePiece arguments
    parser.add_argument("--vocab_size", type=int, default=500, help="Vocabulary size for SentencePiece model")
    parser.add_argument("--spm_model_file", type=str, default=None, help="Path to the pre-trained SentencePiece model file")
    parser.add_argument("--spm_vocab_file", type=str, default=None, help="Path to the SentencePiece vocabulary file")
    parser.add_argument("--skip_tokenization", action="store_true", help="Skip creation of .bin files")

    # Tiktoken arguments
    parser.add_argument("-e", "--tiktoken_encoding",
                       choices=["gpt2", "r50k_base", "p50k_base", "cl100k_base"],
                       default="gpt2", help="Version of tiktoken encoding to utilize")
    parser.add_argument("--additional_tokens_file", type=str, default=None,
                       help="Path to JSON file containing additional special tokens for tiktoken (format: {'token': id})")

    # Char tokenizer arguments
    parser.add_argument("--reuse_chars", action="store_true", help="Reuse character list from meta.pkl")

    # Custom tokenizer arguments
    parser.add_argument("--tokens_file", type=str, default=None, help="Path to the file containing newline-separated tokens for tokenization")
    parser.add_argument("--custom_chars_file", type=str, default=None, help="Path to the file containing custom characters for the tokenizer")
    parser.add_argument("--json_tokens_file", type=str, default=None, help="Path to JSON file containing tokens for json_byte_fallback tokenizer")

    # Audio FFT tokenizer arguments
    parser.add_argument("--audio_fft_sample_rate", type=int, default=16000, help="Target sample rate for audio tokenization")
    parser.add_argument("--audio_fft_n_fft", type=int, default=400, help="FFT window size (Whisper uses 400)")
    parser.add_argument("--audio_fft_hop_length", type=int, default=160, help="FFT hop length (Whisper uses 160)")
    parser.add_argument("--audio_fft_n_mels", type=int, default=80, help="Number of mel bins before grouping")
    parser.add_argument("--audio_fft_channel_splits", type=str, default=None,
                        help="Comma separated mel bin counts per channel (defaults to 5 equal groups)")
    parser.add_argument("--audio_fft_channel_reduction", type=str, default="mean", choices=["mean", "max"],
                        help="Reduction to apply inside each mel group")
    parser.add_argument("--audio_fft_quantization_bits", type=int, default=16,
                        help="Number of bits to quantize normalized log-mel values (<=16)")
    parser.add_argument("--audio_fft_pad_to_samples", type=int, default=None,
                        help="Pad/trim audio to this many samples (Whisper encoder uses 480000 for 30s)")
    parser.add_argument("--audio_fft_output_root", type=str, default=None,
                        help="Optional directory where per-channel outputs will be written (defaults to current folder)")

    # Additional options
    parser.add_argument("-T", "--track_token_counts", action="store_true", help="Track how often each token appears and store in meta.pkl")

    return parser.parse_args()

def save_tokens(ids, output_file, dtype):
    """Save tokenized data to a binary file with progress bar."""
    total = len(ids)
    batch_size = 1024 * 1024  # 1 million tokens per batch
    with open(output_file, 'wb') as f_out:
        for i in tqdm(range(0, total, batch_size), desc=f"Saving {output_file}"):
            batch = ids[i:i+batch_size]
            np.array(batch, dtype=dtype).tofile(f_out)

def main():
    args = parse_arguments()

    # Load training/validation data depending on tokenizer method
    if args.method == "sinewave":
        train_data = None
        val_data = None
    elif args.method == "audio_fft":
        tokenizer = AudioFFTTokenizer(args)
        train_wave = AudioFFTTokenizer._load_audio(
            args.train_input,
            args.audio_fft_sample_rate,
            args.audio_fft_pad_to_samples,
        )
        val_wave = None
        if args.val_input:
            val_wave = AudioFFTTokenizer._load_audio(
                args.val_input,
                args.audio_fft_sample_rate,
                args.audio_fft_pad_to_samples,
            )
    else:
        with open(args.train_input, 'r') as f:
            train_data = f.read()

        if args.val_input:
            with open(args.val_input, 'r') as f:
                val_data = f.read()
        else:
            n = len(train_data)
            train_data, val_data = train_data[:int(n * args.percentage_train)], train_data[int(n * args.percentage_train):]
            if args.percentage_train == 1.0:
                val_data = None

    # Initialize tokenizer based on method
    if args.method == "audio_fft":
        channel_tokens, channel_metas = tokenizer.tokenize(train_wave)
        if args.val_input:
            val_channel_tokens, _ = tokenizer.tokenize(val_wave)
        else:
            val_channel_tokens = {}
            train_channel_tokens = {}
            for name, seq in channel_tokens.items():
                split_idx = int(len(seq) * args.percentage_train)
                train_channel_tokens[name] = seq[:split_idx]
                val_channel_tokens[name] = seq[split_idx:] if args.percentage_train < 1.0 else []
            channel_tokens = train_channel_tokens

        dtype = np.uint16
        output_root = args.audio_fft_output_root or os.getcwd()
        os.makedirs(output_root, exist_ok=True)

        manifest = {
            "tokenizer": "audio_fft",
            "sample_rate": args.audio_fft_sample_rate,
            "n_fft": args.audio_fft_n_fft,
            "hop_length": args.audio_fft_hop_length,
            "n_mels": args.audio_fft_n_mels,
            "quantization_bits": args.audio_fft_quantization_bits,
            "has_validation": bool(args.val_input) or args.percentage_train < 1.0,
            "channels": [],
        }

        write_val = bool(args.val_input) or args.percentage_train < 1.0

        for name in tokenizer.channel_names:
            channel_dir = os.path.join(output_root, name)
            os.makedirs(channel_dir, exist_ok=True)

            train_tokens = channel_tokens[name]
            if len(train_tokens) == 0:
                raise ValueError(f"Channel {name} produced no training tokens")

            val_tokens = val_channel_tokens.get(name, [])
            if write_val and len(val_tokens) == 0:
                # Ensure validation files are non-empty for downstream memmaps.
                val_tokens = [train_tokens[-1]]
                print(f"Warning: validation split for {name} was empty; duplicating last training frame.")

            save_tokens(train_tokens, os.path.join(channel_dir, "train.bin"), dtype)
            if write_val:
                save_tokens(val_tokens, os.path.join(channel_dir, "val.bin"), dtype)

            with open(os.path.join(channel_dir, "meta.pkl"), "wb") as f:
                pickle.dump(channel_metas[name], f)

            manifest["channels"].append({
                "name": name,
                "relative_path": os.path.relpath(channel_dir, output_root),
                "train_tokens": len(train_tokens),
                "val_tokens": len(val_tokens) if write_val else 0,
                "mel_bins": channel_metas[name]["mel_bins"],
            })

        manifest_path = os.path.join(output_root, "audio_fft_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print("Audio FFT tokenization complete. Per-channel data saved to:", output_root)
        print("Manifest written to:", manifest_path)
        return
    elif args.method == "sentencepiece":
        tokenizer = SentencePieceTokenizer(args, input_files=args.train_input)
    elif args.method == "tiktoken":
        tokenizer = TiktokenTokenizer(args)
    elif args.method == "custom":
        tokenizer = CustomTokenizer(args)
    elif args.method == "byte":
        tokenizer = ByteTokenizer(args)
    elif args.method == "char":
        tokenizer = CharTokenizer(args, train_data, val_data)
    elif args.method == "custom_char_byte_fallback":
        tokenizer = CustomCharTokenizerWithByteFallback(args)
    elif args.method == "json_byte_fallback":
        tokenizer = JsonByteTokenizerWithByteFallback(args)
    elif args.method == "sinewave":
        tokenizer = SineWaveTokenizer(args)
    else:
        raise ValueError(f"Unknown tokenization method: {args.method}")

    # Tokenize data
    train_ids = tokenizer.tokenize(train_data)
    if args.method == "tiktoken":
        print(f"[tiktoken] Total train tokens: {tokenizer.last_token_count:,}")
    if args.method == "sinewave" and args.val_input is None:
        split_point = int(len(train_ids) * args.percentage_train)
        val_ids = train_ids[split_point:]
        train_ids = train_ids[:split_point]
    elif val_data is not None:
        val_ids = tokenizer.tokenize(val_data)
        if args.method == "tiktoken":
            print(f"[tiktoken] Total val tokens: {tokenizer.last_token_count:,}")
    else:
        val_ids = None

    # Determine dtype based on vocabulary size from meta.pkl
    if args.method == "sinewave":
        dtype = np.uint16
    else:
        with open("meta.pkl", "rb") as f:
            meta = pickle.load(f)
        vocab_size = meta["vocab_size"]
        dtype = np.uint32 if vocab_size > 65535 else np.uint16

    # Ensure output directories exist if paths include folders
    for output_path in [args.train_output, args.val_output]:
        if output_path:
            out_dir = os.path.dirname(output_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

    # Save tokenized data
    save_tokens(train_ids, args.train_output, dtype)
    if val_ids is not None:
        save_tokens(val_ids, args.val_output, dtype)

    if args.method == "sinewave":
        meta = {
            "tokenizer": "sinewave",
            "vocab_size": 256,
            "sine_period": args.sine_period,
            "sine_points_per_period": args.sine_points_per_period,
            "sine_num_periods": args.sine_num_periods,
            "sine_amplitude": args.sine_amplitude,
        }
        with open("meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    # Save additional metadata for tiktoken if needed
    if args.method == "tiktoken" and args.additional_tokens_file:
        with open(args.additional_tokens_file, 'r') as f:
            additional_tokens = json.load(f)
        with open("meta.pkl", "rb") as f:
            meta = pickle.load(f)
        meta.update({
            "has_additional_tokens": True,
            "special_tokens": additional_tokens,
            "tokenizer": "tiktoken",
            "tiktoken_encoding": args.tiktoken_encoding
        })
        with open("meta.pkl", "wb") as f:
            pickle.dump(meta, f)

if __name__ == "__main__":
    main()

