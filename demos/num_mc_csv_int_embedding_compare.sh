#!/usr/bin/env bash
# Compare numerical embedding variants on integer-quantized CSV multicontext data.
#
# Variants compared:
#   mlp             - standard MLP embedding (baseline)
#   scaled_vector   - single shared learned vector scaled by scalar
#   learned_vector  - per-channel randomly-initialized learned vector scaled by
#                     scalar with a settable global attenuation coefficient
#   log_scaled_vector - like learned_vector but scalar is first passed through
#                     an activation (default: log) before scaling the vector
#
# Usage:
#   ./demos/num_mc_csv_int_embedding_compare.sh [path/to/input.csv]
#
# The first positional argument overrides the default CSV source.
# Plotly HTML outputs land in out/num_emb_compare/<variant>/samples.html

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CSV_INPUT="${1:-data/csv_num_mc_int/input.csv}"

# ---------------------------------------------------------------------------
# 1. Prepare per-column integer datasets (shared across all variants)
# ---------------------------------------------------------------------------
data/csv_num_mc_int/get_datasets.sh "$CSV_INPUT" \
  --output_root csv_num_mc_int \
  --train_ratio 0.9 \
  --column-transform bpm:-30:250 \
  --column-transform spo2:-60:500 \
  --column-transform movement:0:200

# ---------------------------------------------------------------------------
# 2. Common training hyper-parameters (same for all variants so results are
#    directly comparable)
# ---------------------------------------------------------------------------
COMMON_TRAIN_ARGS=(
  --training_mode multicontext
  --dataset csv_num_mc_int/bpm
  --multicontext
  --multicontext_datasets
    csv_num_mc_int/bpm
    csv_num_mc_int/spo2
    csv_num_mc_int/movement
  --numerical_multicontext
  --numerical_multicontext_input_format scalar
  --numerical_output_variant mlp
  --numerical_mlp_hidden_dims 128 128 128
  --use_qk_norm
  --use_qk_norm_scale
  --use_rotary_embeddings
  --no-use_abs_pos_embeddings
  --attention_variant infinite
  --use_concat_heads
  --n_layer 6
  --n_head 3
  --n_qk_head_dim 128
  --n_v_head_dim 128
  --n_embd 384
  --block_size 256
  --batch_size 32
  --max_iters 2000
  --eval_interval 200
  --eval_iters 50
  --learning_rate 3e-4
  --dtype bfloat16
  --compile
)

COMMON_SAMPLE_ARGS=(
  --multicontext
  --multicontext_datasets
    csv_num_mc_int/bpm
    csv_num_mc_int/spo2
    csv_num_mc_int/movement
  --multicontext_start "11000" "18688" "800"
  --numerical_multicontext_plotly
  --max_new_tokens 256
  --num_samples 1
)

# ---------------------------------------------------------------------------
# 3. mlp  (baseline)
# ---------------------------------------------------------------------------
echo "=== Variant: mlp ==="
python3 train.py \
  "${COMMON_TRAIN_ARGS[@]}" \
  --numerical_embedding_variant mlp \
  --out_dir out/num_emb_compare/mlp

python3 sample.py \
  "${COMMON_SAMPLE_ARGS[@]}" \
  --out_dir out/num_emb_compare/mlp \
  --numerical_multicontext_plotly_file out/num_emb_compare/mlp/samples.html

# ---------------------------------------------------------------------------
# 4. scaled_vector  (existing single shared vector)
# ---------------------------------------------------------------------------
echo "=== Variant: scaled_vector ==="
python3 train.py \
  "${COMMON_TRAIN_ARGS[@]}" \
  --numerical_embedding_variant scaled_vector \
  --out_dir out/num_emb_compare/scaled_vector

python3 sample.py \
  "${COMMON_SAMPLE_ARGS[@]}" \
  --out_dir out/num_emb_compare/scaled_vector \
  --numerical_multicontext_plotly_file out/num_emb_compare/scaled_vector/samples.html

# ---------------------------------------------------------------------------
# 5. learned_vector  (per-channel random-init vector, attenuation=1.0)
# ---------------------------------------------------------------------------
echo "=== Variant: learned_vector (attn_coeff=1.0) ==="
python3 train.py \
  "${COMMON_TRAIN_ARGS[@]}" \
  --numerical_embedding_variant learned_vector \
  --numerical_learned_vector_attn_coeff 1.0 \
  --out_dir out/num_emb_compare/learned_vector

python3 sample.py \
  "${COMMON_SAMPLE_ARGS[@]}" \
  --out_dir out/num_emb_compare/learned_vector \
  --numerical_multicontext_plotly_file out/num_emb_compare/learned_vector/samples.html

# ---------------------------------------------------------------------------
# 6. learned_vector  (attenuation=0.1 to show effect of the coefficient)
# ---------------------------------------------------------------------------
echo "=== Variant: learned_vector (attn_coeff=0.1) ==="
python3 train.py \
  "${COMMON_TRAIN_ARGS[@]}" \
  --numerical_embedding_variant learned_vector \
  --numerical_learned_vector_attn_coeff 0.1 \
  --out_dir out/num_emb_compare/learned_vector_attn01

python3 sample.py \
  "${COMMON_SAMPLE_ARGS[@]}" \
  --out_dir out/num_emb_compare/learned_vector_attn01 \
  --numerical_multicontext_plotly_file out/num_emb_compare/learned_vector_attn01/samples.html

# ---------------------------------------------------------------------------
# 7. log_scaled_vector  (log activation, default)
# ---------------------------------------------------------------------------
echo "=== Variant: log_scaled_vector (activation=log) ==="
python3 train.py \
  "${COMMON_TRAIN_ARGS[@]}" \
  --numerical_embedding_variant log_scaled_vector \
  --numerical_log_vector_activation log \
  --numerical_learned_vector_attn_coeff 1.0 \
  --out_dir out/num_emb_compare/log_scaled_vector

python3 sample.py \
  "${COMMON_SAMPLE_ARGS[@]}" \
  --out_dir out/num_emb_compare/log_scaled_vector \
  --numerical_multicontext_plotly_file out/num_emb_compare/log_scaled_vector/samples.html

# ---------------------------------------------------------------------------
# 8. log_scaled_vector  (log1p activation for numerically safer handling of
#    values near zero)
# ---------------------------------------------------------------------------
echo "=== Variant: log_scaled_vector (activation=log1p) ==="
python3 train.py \
  "${COMMON_TRAIN_ARGS[@]}" \
  --numerical_embedding_variant log_scaled_vector \
  --numerical_log_vector_activation log1p \
  --numerical_learned_vector_attn_coeff 1.0 \
  --out_dir out/num_emb_compare/log1p_scaled_vector

python3 sample.py \
  "${COMMON_SAMPLE_ARGS[@]}" \
  --out_dir out/num_emb_compare/log1p_scaled_vector \
  --numerical_multicontext_plotly_file out/num_emb_compare/log1p_scaled_vector/samples.html

# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------
echo ""
echo "All variants finished.  Plotly reports:"
for variant in mlp scaled_vector learned_vector learned_vector_attn01 \
               log_scaled_vector log1p_scaled_vector; do
  echo "  out/num_emb_compare/${variant}/samples.html"
done
