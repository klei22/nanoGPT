#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Run sweep with longest-bytes trimming strategy
python latin_punct_router_eval.py \
  --model_name google/gemma-3-270m-it \
  --route_mode latin_punct_only \
  --latin_trim_sweep \
  --latin_trim_strategy longest_bytes \
  --latin_trim_sweep_max 80 \
  --latin_trim_sweep_step 10 \
  --split "validation[:1%]" \
  --max_samples 100 \
  --max_target_tokens 64 \
  --example_split "validation[:20]" \
  --sweep_examples 10 \
  --example_max_new_tokens 64 \
  --report_dir latin_trim_reports_longest_bytes \
  --byte_fallback

# Run sweep with highest-id trimming strategy
python latin_punct_router_eval.py \
  --model_name google/gemma-3-270m-it \
  --route_mode latin_punct_only \
  --latin_trim_sweep \
  --latin_trim_strategy highest_id \
  --latin_trim_sweep_max 80 \
  --latin_trim_sweep_step 10 \
  --split "validation[:1%]" \
  --max_samples 100 \
  --max_target_tokens 64 \
  --example_split "validation[:20]" \
  --sweep_examples 10 \
  --example_max_new_tokens 64 \
  --report_dir latin_trim_reports_highest_id \
  --byte_fallback

# Run quantization sweep on 100% latin+punct(+byte) candidate set
python latin_punct_router_eval.py \
  --model_name google/gemma-3-270m-it \
  --route_mode latin_punct_only \
  --quantization_sweep \
  --quant_group_size 32 \
  --split "validation[:1%]" \
  --max_samples 100 \
  --max_target_tokens 64 \
  --quant_report_dir quantization_reports \
  --byte_fallback

# Build combined graph: routed(longest-bytes) + routed(highest-id) + full LM head
python - <<'PY'
import csv
import matplotlib.pyplot as plt

def read_csv(path):
    xs, full, routed = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(int(float(row["latin_trim_percent"])))
            full.append(float(row["top1_full_percent"]))
            routed.append(float(row["top1_routed_percent"]))
    return xs, full, routed

xs_a, full_a, routed_a = read_csv("latin_trim_reports_longest_bytes/latin_trim_sweep.csv")
xs_b, full_b, routed_b = read_csv("latin_trim_reports_highest_id/latin_trim_sweep.csv")

if xs_a != xs_b:
    raise RuntimeError(f"Mismatched sweep x-values: {xs_a} vs {xs_b}")

quant_labels, quant_vals = [], []
with open("quantization_reports/quantization_sweep.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        bits = int(row["bits"])
        mode = row["mode"]
        label = "baseline" if bits == 0 else f"{mode}-{bits}b"
        quant_labels.append(label)
        quant_vals.append(float(row["top1_percent"]))

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
ax0, ax1 = axes
ax0.plot(xs_a, full_a, marker="o", label="Full LM head")
ax0.plot(xs_a, routed_a, marker="o", label="Routed (longest_bytes)")
ax0.plot(xs_b, routed_b, marker="o", label="Routed (highest_id)")
ax0.set_xlabel("Latin trim percent")
ax0.set_ylabel("Top-1 accuracy (%)")
ax0.set_title("Trim strategies vs Full LM head")
ax0.grid(True, alpha=0.3)
ax0.legend()

ax1.bar(range(len(quant_labels)), quant_vals)
ax1.set_xticks(range(len(quant_labels)))
ax1.set_xticklabels(quant_labels, rotation=70, ha="right")
ax1.set_ylabel("Top-1 accuracy (%)")
ax1.set_title("Quantization sweep (latin+punct+byte)")

plt.tight_layout()
out_path = "latin_trim_reports_combined_accuracy.png"
plt.savefig(out_path, dpi=180)
print(f"Wrote combined graph: {out_path}")
PY
