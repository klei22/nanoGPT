#!/usr/bin/env bash
set -euo pipefail

# Run sweep with longest-bytes trimming strategy
python huggingface_model/gemma/270M/latin_punct_router_eval.py \
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
python huggingface_model/gemma/270M/latin_punct_router_eval.py \
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

plt.figure(figsize=(8, 5))
plt.plot(xs_a, full_a, marker="o", label="Full LM head")
plt.plot(xs_a, routed_a, marker="o", label="Routed (longest_bytes)")
plt.plot(xs_b, routed_b, marker="o", label="Routed (highest_id)")
plt.xlabel("Latin trim percent")
plt.ylabel("Top-1 accuracy (%)")
plt.title("Latin trim strategies vs Full LM head")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
out_path = "latin_trim_reports_combined_accuracy.png"
plt.savefig(out_path, dpi=180)
print(f"Wrote combined graph: {out_path}")
PY
