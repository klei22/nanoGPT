# !/bin/bash

# Set strict error handling
set -euo pipefail

# Install python dependencies for Hugging face
pip install -U "huggingface_hub[cli]"

# Authentication with Hugging Face
# Replace with your hugging face tokens
##### You can find and create your own tokens here: https://huggingface.co/settings/tokens ######
##### "Token Type" of "Read" is recommended. ########
HF_TOKEN=""

# Authenticate with hugging face
echo "Authenticating with Hugging Face..."
huggingface-cli login --token "${HF_TOKEN}"

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

url="https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0"
out_dir="transcription"

if [[ ! -d "${out_dir}" ]]; then
  mkdir -p "${out_dir}"
fi

# Download transcription files under "transcription" directory.
pushd "${out_dir}"
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "dev.tsv" "${url}/resolve/main/transcript/zh-CN/dev.tsv?download=true"
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "other.tsv" "${url}/resolve/main/transcript/zh-CN/other.tsv?download=true"
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "test.tsv" "${url}/resolve/main/transcript/zh-CN/test.tsv?download=true"
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "train.tsv" "${url}/resolve/main/transcript/zh-CN/train.tsv?download=true"
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "validated.tsv" "${url}/resolve/main/transcript/zh-CN/validated.tsv?download=true"

echo "transcripts downloaded and saved to transcription."
popd

audio_zip_dir="zh_tar_audio"
audio_dir="zh_audio"

if [[ ! -d "${audio_zip_dir}" ]]; then
  mkdir -p "${audio_zip_dir}"
fi

# Download audio files under "zh_audio" directory.
pushd "${audio_zip_dir}"
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "zh-CN_dev_0.tar" "${url}/resolve/main/audio/zh-CN/dev/zh-CN_dev_0.tar?download=true"
for i in $(seq 0 14); do
    wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "zh-CN_other_${i}.tar" "${url}/resolve/main/audio/zh-CN/other/zh-CN_other_${i}.tar?download=true"
done
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "zh-CN_test_0.tar" "${url}/resolve/main/audio/zh-CN/test/zh-CN_test_0.tar?download=true"
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "zh-CN_train_0.tar" "${url}/resolve/main/audio/zh-CN/train/zh-CN_train_0.tar?download=true"

# Create directory to store all the audio files
if [[ ! -d "${audio_dir}" ]]; then
    mkdir -p "${audio_dir}"
fi

# Loop through each .tar file and extract them
for tarfile in *.tar; do
    # Check if the .tar file exists (handles the case where no .tar files are present)
    if [ -f "$tarfile" ]; then
        echo "Extracting $tarfile..."
        tar --strip-components=1 -xvf "$tarfile" -C "${audio_dir}" > /dev/null
    fi
done
popd

json_dir="json_outs"

if [[ ! -d "${json_dir}" ]]; then
  mkdir -p "${json_dir}"
fi

# Run program to get snac, text combined json file
for tsvfile in "$out_dir"/*.tsv; do
    # Check if the .tsv file exists (handles the case where no .tsv files are present)
    if [ -f "$tsvfile" ]; then
        echo "Processing $tsvfile..."
        # Get the filename without the extension for output filename
        filename=$(basename "${tsvfile%.tsv}")
        output_file="$json_dir/$filename.json"
        python3 snac_text_zh.py "$audio_zip_dir/$audio_dir" "$tsvfile" "$output_file"
    fi
done

echo "All .tsv files have been processed."
