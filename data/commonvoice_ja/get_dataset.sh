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
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "dev.tsv" "${url}/resolve/main/transcript/ja/dev.tsv?download=true"
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "invalidated.tsv" "${url}/resolve/main/transcript/ja/validated.tsv?download=true"
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "other.tsv" "${url}/resolve/main/transcript/ja/other.tsv?download=true"
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "test.tsv" "${url}/resolve/main/transcript/ja/test.tsv?download=true"
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "train.tsv" "${url}/resolve/main/transcript/ja/train.tsv?download=true"
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "validated.tsv" "${url}/resolve/main/transcript/ja/validated.tsv?download=true"

echo "transcripts downloaded and saved to transcription."
popd

# Run program to convert tsv into json format.
output_file="ja_transcription.json"
for tsvfile in "$out_dir"/*.tsv; do
    # Check if the .tsv file exists (handles the case where no .tsv files are present)
    if [ -f "$tsvfile" ]; then
        echo "Processing $tsvfile..."
        # Get the filename without the extension for output filename
        filename=$(basename "${tsvfile%.tsv}")
        python3 utils/tsv_to_json_cv.py "$tsvfile" "$output_file"
    fi
done

echo "All .tsv files have been processed."

# Run program to convert sentences into IPA format.
output_ipa="ja_ipa.txt"
echo "Converting sentences to IPA..."
python3 utils/ja2ipa.py "$output_file" "$output_ipa"

echo "IPA conversion finished."

# Tokenization step to create train.bin and val.bin files.
python3 prepare.py -t "$output_ipa" --method char
