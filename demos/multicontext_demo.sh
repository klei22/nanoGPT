#!/bin/bash
# multicontext_demo.sh


# Check for dependencies
check_spacy() {
  python3 - <<'PY'
import sys

try:
    import spacy
    spacy.load("en_core_web_sm", disable=["parser", "ner"])
except Exception as exc:
    print(f"spaCy check failed: {exc}", file=sys.stderr)
    sys.exit(1)
PY
}

ensure_spacy() {
  if check_spacy; then
    return
  fi

  echo "spaCy and its English model are required for part-of-speech conversion." >&2
  echo "To install them, run:" >&2
  echo "  python3 -m pip install --upgrade pip" >&2
  echo "  python3 -m pip install spacy" >&2
  echo "  python3 -m spacy download en_core_web_sm" >&2
  exit 1
}

ensure_spacy

pushd data/shakespeare_char
bash get_dataset.sh
popd

pushd data/shakespeare_char_case_map/
bash get_dataset.sh
popd

pushd data/shakespeare_char_lowercase/
bash get_dataset.sh
popd

pushd data/shakespeare_char_cvp/
bash get_dataset.sh
popd

pushd data/shakespeare_char_in_word_position/
bash get_dataset.sh
popd

pushd data/shakespeare_char_part_of_speech/
bash get_dataset.sh
popd

pushd data/shakespeare_char_since_newline/
bash get_dataset.sh
popd

pushd data/shakespeare_char_newlines_mod/
bash get_dataset.sh
popd

 python3 train.py \
   --training_mode multicontext \
   --multicontext \
   --multicontext_datasets \
       shakespeare_char \
       shakespeare_char_case_map \
       shakespeare_char_lowercase \
       shakespeare_char_cvp \
       shakespeare_char_in_word_position \
       shakespeare_char_part_of_speech \
       shakespeare_char_since_newline \
       shakespeare_char_newlines_mod \
    --max_iters 2000 \
    --dropout 0.2 \
    --top_k 1 \
    --sample_each_eval \
    --use_qk_norm \
    --use_qk_norm_scale \
    --use_rotary_embeddings \
    --no-use_abs_pos_embeddings \
    --out_dir ./out_mc_shakespeare \
    --compile

python3 sample.py \
  --out_dir ./out_mc_shakespeare \
  --multicontext \
  --multicontext_datasets \
    shakespeare_char \
    shakespeare_char_case_map \
    shakespeare_char_lowercase \
    shakespeare_char_cvp \
    shakespeare_char_in_word_position \
    shakespeare_char_part_of_speech \
    shakespeare_char_since_newline \
    shakespeare_char_newlines_mod \
  --multicontext_start \
    "But " \
    "ULL_" \
    "but " \
    "323_" \
    "124_" \
    "fff " \
    "1234" \
    "1111" \
  --max_new_tokens 512 \
  --top_k 1 \
  --num_samples 3

