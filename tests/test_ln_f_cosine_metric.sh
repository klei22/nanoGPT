#!/bin/bash
# test_ln_f_cosine_metric.sh
# Runs a tiny experiment and checks ln_f cosine similarity metric is logged.
set -e
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$script_dir/.."

log_file="exploration_logs/test_ln_f_cosine_config.yaml"
out_dir="results/lnf_cos_test"
# clean previous runs
rm -f "$log_file"
rm -rf "$out_dir"*


dataset="shakespeare_char"
# ensure dataset is present
bash "data/${dataset}/get_dataset.sh"

config_file="tests/test_ln_f_cosine_config.yaml"

python3 optimization_and_search/run_experiments.py -c "$config_file" --config_format yaml

python3 - <<'PY'
import yaml, pathlib, math
log_path = pathlib.Path('exploration_logs/test_ln_f_cosine_config.yaml')
entries = list(yaml.safe_load_all(log_path.read_text()))
assert entries, 'no entries found in log file'
cos = entries[-1].get('avg_ln_f_cosine')
assert cos is not None, 'avg_ln_f_cosine missing'
assert not math.isnan(cos), 'avg_ln_f_cosine is NaN'
print('avg_ln_f_cosine:', cos)
PY
