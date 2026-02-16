import argparse
import os
import subprocess

from tqdm import tqdm

def batch_prepare(
    input_dir,
    train_output,
    val_output,
    prepare_script,
    tokenizer,
    spm_model_file,
    spm_vocab_file,
    char_bpe_vocab_path,
    json_tokens_file,
    train_ratio=0.9,
):
    files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.txt')])
    files = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
    print(files)
    num_train = int(len(files) * train_ratio)

    if tokenizer == "tiktoken":
        # Train files
        for file in tqdm(files[:num_train], desc="Processing training files"):
            subprocess.run(['python3', prepare_script, '--train_input', file, '--train_output', file.replace('.txt', '.bin'), '-p 1.0'])

        # Validation files
        for file in tqdm(files[num_train:], desc="Processing validation files"):
            subprocess.run(['python3', prepare_script, '--train_input', file, '--train_output', file.replace('.txt', '.bin'), '-p 1.0'])

        # Combine bins
        combine_bins(files[:num_train], train_output, desc="Combining training files")
        combine_bins(files[num_train:], val_output, desc="Combining validation files")
        print(f"Created {train_output} and {val_output}")
    elif tokenizer == "sentencepiece":
        # Train files
        for file in tqdm(files[:num_train], desc="Processing training files"):
            subprocess.run(['python3', prepare_script, '--method', tokenizer, '--spm_model_file', spm_model_file, '--spm_vocab_file', spm_vocab_file, '--train_input', file, '--train_output', file.replace('.txt', '.bin'), '-p 1.0'])

        # Validation files
        for file in tqdm(files[num_train:], desc="Processing validation files"):
            subprocess.run(['python3', prepare_script, '--method', tokenizer, '--spm_model_file', spm_model_file, '--spm_vocab_file', spm_vocab_file, '--train_input', file, '--train_output', file.replace('.txt', '.bin'), '-p 1.0'])

        # Combine bins
        combine_bins(files[:num_train], train_output, desc="Combining training files")
        combine_bins(files[num_train:], val_output, desc="Combining validation files")
        print(f"Created {train_output} and {val_output}")
    elif tokenizer == "char":
        # Train files
        for file in tqdm(files[:num_train], desc="Processing training files"):
            subprocess.run(['python3', prepare_script, '--method', tokenizer, '--reuse_chars', '--train_input', file, '--train_output', file.replace('.txt', '.bin'), '-p 1.0'])

        # Validation files
        for file in tqdm(files[num_train:], desc="Processing validation files"):
            subprocess.run(['python3', prepare_script, '--method', tokenizer, '--reuse_chars', '--train_input', file, '--train_output', file.replace('.txt', '.bin'), '-p 1.0'])

        # Combine bins
        combine_bins(files[:num_train], train_output, desc="Combining training files")
        combine_bins(files[num_train:], val_output, desc="Combining validation files")
        print(f"Created {train_output} and {val_output}")
    elif tokenizer == "char_bpe":
        if not char_bpe_vocab_path:
            raise ValueError(
                "--char_bpe_vocab_path is required when --tokenizer char_bpe. "
                "Provide the path to a previously generated char_bpe meta.pkl file."
            )

        # Train files
        for file in tqdm(files[:num_train], desc="Processing training files"):
            subprocess.run([
                'python3',
                prepare_script,
                '--method',
                tokenizer,
                '--char_bpe_vocab_path',
                char_bpe_vocab_path,
                '--train_input',
                file,
                '--train_output',
                file.replace('.txt', '.bin'),
                '-p',
                '1.0',
            ])

        # Validation files
        for file in tqdm(files[num_train:], desc="Processing validation files"):
            subprocess.run([
                'python3',
                prepare_script,
                '--method',
                tokenizer,
                '--char_bpe_vocab_path',
                char_bpe_vocab_path,
                '--train_input',
                file,
                '--train_output',
                file.replace('.txt', '.bin'),
                '-p',
                '1.0',
            ])

        # Combine bins
        combine_bins(files[:num_train], train_output, desc="Combining training files")
        combine_bins(files[num_train:], val_output, desc="Combining validation files")
        print(f"Created {train_output} and {val_output}")
    elif tokenizer == "json_byte_fallback":
        if not json_tokens_file:
            raise ValueError(
                "--json_tokens_file is required when --tokenizer json_byte_fallback. "
                "Provide a path to a JSON array of tokens."
            )

        # Train files
        for file in tqdm(files[:num_train], desc="Processing training files"):
            subprocess.run([
                'python3',
                prepare_script,
                '--method',
                tokenizer,
                '--json_tokens_file',
                json_tokens_file,
                '--train_input',
                file,
                '--train_output',
                file.replace('.txt', '.bin'),
                '-p',
                '1.0',
            ])

        # Validation files
        for file in tqdm(files[num_train:], desc="Processing validation files"):
            subprocess.run([
                'python3',
                prepare_script,
                '--method',
                tokenizer,
                '--json_tokens_file',
                json_tokens_file,
                '--train_input',
                file,
                '--train_output',
                file.replace('.txt', '.bin'),
                '-p',
                '1.0',
            ])

        # Combine bins
        combine_bins(files[:num_train], train_output, desc="Combining training files")
        combine_bins(files[num_train:], val_output, desc="Combining validation files")
        print(f"Created {train_output} and {val_output}")

    else:
        print(f"tokenizer {tokenizer} not currently supported")
        return

def combine_bins(files, output_file, desc="Combining files"):
    with open(output_file, 'wb') as fout:
        for file in tqdm(files, desc=desc):
            bin_file = file.replace('.txt', '.bin')
            if os.path.exists(bin_file):
                with open(bin_file, 'rb') as fin:
                    while chunk := fin.read(1024):  # Read in chunks of 1024 bytes
                        fout.write(chunk)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training and validation binary files from divided text files.")
    parser.add_argument("--input_dir", type=str, default="partitioned_file", help="Directory containing the divided files.")
    parser.add_argument("--train_output", type=str, default="train.bin", help="Output binary file for training data.")
    parser.add_argument("--val_output", type=str, default="val.bin", help="Output binary file for validation data.")
    parser.add_argument("--prepare_script", type=str, default="prepare.py", help="Path to the prepare.py script.")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ratio of training data to total data.")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer method.")
    parser.add_argument("--spm_model", type=str, default="trained_spm_model.model", help="SPM model file.")
    parser.add_argument("--spm_vocab", type=str, default="trained_spm_model.vocab", help="SPM vocab file.")
    parser.add_argument(
        "--char_bpe_vocab_path",
        type=str,
        default=None,
        help="Path to an existing char_bpe meta.pkl used to reuse a prebuilt vocab.",
    )
    parser.add_argument(
        "--json_tokens_file",
        type=str,
        default=None,
        help="Path to a JSON token list used by json_byte_fallback tokenization.",
    )

    args = parser.parse_args()
    batch_prepare(
        args.input_dir,
        args.train_output,
        args.val_output,
        args.prepare_script,
        args.tokenizer,
        args.spm_model,
        args.spm_vocab,
        args.char_bpe_vocab_path,
        args.json_tokens_file,
        train_ratio=args.train_ratio,
    )
