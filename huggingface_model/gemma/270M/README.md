# Gemma 270M workflows

This folder includes scripts for training Gemma 270M from scratch, fine-tuning, and
experimenting with LM head acceleration using Johnson-Lindenstrauss (JL) projection.

## Loading a pre-trained checkpoint

Use the Hugging Face model hub (or a local checkpoint directory) as the `model_name`:

```bash
python huggingface_model/gemma/270M/jl_head_eval.py \
  --model_name google/gemma-3-270m \
  --dataset_split "train[:1%]" \
  --fineweb_subset sample-10BT
```

The scripts rely on `AutoModelForCausalLM.from_pretrained(...)`, so you can point
`--model_name` at any local checkpoint path created by `train_from_scratch.py` or
`finetune.py`.

## JL-projected LM head evaluation

`jl_head_eval.py` runs a two-stage LM head evaluation:

1. Project hidden states and LM head weights into a lower `target_dimension` using
   a JL random matrix.
2. Compute approximate logits, select the top-`n` candidate tokens, and then compute
   exact logits only for that subset.

It reports the average absolute token-id difference between the estimated top-k
tokens and the exact top-k tokens, and plots a heatmap for a sweep of `top_n`
values and `target_dimension` sizes.

**How `avg_id_delta` is computed**

For each token position, the script compares the top-`k` token IDs from the
exact logits against the top-`k` token IDs from the JL-approximated logits. It
sorts both ID lists, computes the elementwise absolute difference between the
sorted IDs, and averages those differences across all token positions. It also
tracks how many IDs changed when comparing the two top-`k` sets (order does not
matter), and summarizes that count with the same mean/median/min/max/std stats.

Example (top-`k` = 5):

* Exact top-`k` IDs: `[2, 7, 11, 19, 42]`
* Estimated top-`k` IDs: `[3, 8, 11, 21, 40]`
* Absolute differences: `|2-3|, |7-8|, |11-11|, |19-21|, |42-40|`
* Average: `(1 + 1 + 0 + 2 + 2) / 5 = 1.2`

```bash
python huggingface_model/gemma/270M/jl_head_eval.py \
  --model_name google/gemma-3-270m \
  --dataset_split "train[:1%]" \
  --fineweb_subset sample-10BT \
  --eval_tokens 1000 \
  --top_k 65 \
  --top_n_values 1000,2000,5000,10000 \
  --target_dimensions 500,400,300,200,100 \
  --projection orthonormal \
  --approx_logits_device cpu \
  --approx_logits_dtype float16 \
  --approx_chunk_size 5000 \
  --exact_chunk_size 5000 \
  --annotate_stats true \
  --output_dir jl_eval_outputs
```

## Latin/punctuation/other manual router (OPUS-100 en-es)

`latin_punct_router_eval.py` builds three token groups from the Gemma vocabulary:

1. Tokens whose decoded UTF text contains at least one Latin-script codepoint.
2. Tokens that are punctuation/layout-only (Unicode punctuation plus common English/Spanish punctuation symbols, and whitespace/layout tokens like tabs/newlines).
3. Everything else (including byte/special tokens).

It then:

- prints a raw-count and percentage table for all three groups,
- computes one average LM-head vector per group,
- L2-normalizes those three vectors into unit routing prototypes,
- routes each decoding step by dot product of final-layer hidden state with the 3 prototypes,
- and performs next-token scoring only within the routed token subset.
- reports side-by-side teacher-forced token accuracy with and without routing.
- prints validation-set example translations before (full LM head) and after (routed LM head).
- color-highlights the actually generated continuation segment (excluding the fixed multi-shot prompt context).
- uses a fixed 3-shot English→Spanish prompt template before each evaluated/generated example.
- can optionally freeze the model and train 3 route scalars (initialized to 1.0) for the 3 prototypes on OPUS-100 `en-es`.

Example:

```bash
python huggingface_model/gemma/270M/latin_punct_router_eval.py \
  --model_name google/gemma-3-270m-it \
  --split "train[:1%]" \
  --max_samples 100 \
  --max_target_tokens 64 \
  --example_split "validation[:20]" \
  --num_examples 3 \
  --example_max_new_tokens 64 \
  --route_mode three_way \
  --byte_fallback \
  --device cuda
```

Notes:

- Evaluation is teacher-forced next-token prediction over OPUS-100 `en-es` translation targets.
- The script uses cosine-style scoring (unit-normalized hidden state and LM-head rows) for both routing and routed token selection.


Alternative 2-way routing (latin+punct vs other):

```bash
python huggingface_model/gemma/270M/latin_punct_router_eval.py \
  --route_mode latin_punct_vs_other \
  --byte_fallback
```

Use `--no-byte_fallback` to disable unioning byte tokens into non-`other` candidate sets.

Alternative 2-way routing that never selects `other` (latin vs punct only):

```bash
python huggingface_model/gemma/270M/latin_punct_router_eval.py \
  --route_mode latin_vs_punct_only \
  --byte_fallback
```

Single-bucket latin+punct mode (no routing decision; only score in latin+punct section):

```bash
python huggingface_model/gemma/270M/latin_punct_router_eval.py \
  --route_mode latin_punct_only \
  --byte_fallback
```

Train only the three route scalars (model frozen) for `three_way` mode:

```bash
python huggingface_model/gemma/270M/latin_punct_router_eval.py \
  --route_mode three_way \
  --train_route_scales \
  --train_split "train[:1%]" \
  --train_max_samples 100 \
  --train_epochs 1 \
  --train_lr 1e-2
```

Interactive chat mode (type your own English sentence and compare full vs routed translation):

```bash
python huggingface_model/gemma/270M/latin_punct_router_eval.py \
  --chat_mode \
  --route_mode three_way \
  --example_max_new_tokens 64
```

In chat mode, user input is color-highlighted and generated continuations are also highlighted in the printed full outputs.

Latin-trim sweep report mode (trim longest UTF-8 latin tokens from 0%..80% in 10% steps, emit report + graph):

```bash
bash huggingface_model/gemma/270M/demo_latin_trim_sweep.sh
```

The demo runs both trim strategies (`longest_bytes`, `highest_id`) and then writes a combined comparison plot:

- `latin_trim_reports_combined_accuracy.png` (full LM head + both routed strategies).
- The same demo also runs a quantization sweep (8/6/5/4/3 bits; vector/group32; symmetric/asymmetric) on the 100% latin+punct(+byte) candidate set and includes it in the combined plot.

Direct CLI equivalent:

```bash
python huggingface_model/gemma/270M/latin_punct_router_eval.py \
  --route_mode latin_punct_only \
  --latin_trim_sweep \
  --latin_trim_strategy longest_bytes \
  --latin_trim_sweep_max 80 \
  --latin_trim_sweep_step 10 \
  --report_dir latin_trim_reports
```

This writes:

- `latin_trim_reports/latin_trim_sweep.csv`
- `latin_trim_reports/latin_trim_sweep_accuracy.png`
- `latin_trim_reports/latin_trim_sweep_report.txt`
- `latin_trim_reports/latin_trim_00.txt`, `latin_trim_reports/latin_trim_10.txt`, ... per-percent details
- `latin_trim_reports/latin_trim_00_latin_tokens.json`, `latin_trim_reports/latin_trim_10_latin_tokens.json`, ... per-percent latin token arrays (`id`, `token`, `length_bytes`, sorted by descending `length_bytes`)

The sweep also computes an additional score: average ASCII-string difference (via sequence ratio) between prediction and reference before the first newline.

You can also run a single trim without sweep via `--latin_trim_percent <pct>`.

Trimming strategy can be selected with `--latin_trim_strategy`:

- `longest_bytes` (default): trim by largest UTF-8 token byte length first.
- `highest_id`: trim by highest token id first.

Standalone quantization sweep:

```bash
python huggingface_model/gemma/270M/latin_punct_router_eval.py \
  --route_mode latin_punct_only \
  --quantization_sweep \
  --quant_group_size 32 \
  --quant_report_dir quantization_reports
```
