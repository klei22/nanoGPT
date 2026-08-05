"""Per-token loss/count reporting kept deliberately separate from TensorBoard."""

import csv
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist


class PerTokenMetrics:
    """Accumulate training-token exposure and export evaluation snapshots."""

    DETAIL_FIELDS = (
        "iteration", "dataset", "token_id", "train_loss", "train_eval_count",
        "val_loss", "val_eval_count", "training_seen_count",
    )

    def __init__(self, output_dir, vocab_sizes, initial_seen=None):
        self.output_dir = output_dir
        self.vocab_sizes = dict(vocab_sizes)
        self.seen = {
            name: np.zeros(size, dtype=np.int64) for name, size in self.vocab_sizes.items()
        }
        self._local_seen = {}
        if initial_seen:
            self.load_state_dict(initial_seen)
        self.pending = {}
        os.makedirs(output_dir, exist_ok=True)
        self.detail_path = os.path.join(output_dir, "per_token_metrics.csv")
        self.summary_path = os.path.join(output_dir, "per_token_summary.csv")
        self.plot_path = os.path.join(output_dir, "per_token_metrics.html")

    def count_training_batch(self, dataset, targets):
        """Accumulate on-device; defer the small count-vector transfer until evaluation."""
        vocab_size = self.vocab_sizes[dataset]
        values = targets.detach().reshape(-1).to(dtype=torch.long)
        values = values[(values >= 0) & (values < vocab_size)]
        counts = torch.bincount(values, minlength=vocab_size)[:vocab_size]
        pending = self._local_seen.get(dataset)
        if pending is None or pending.device != counts.device:
            if pending is not None:
                self.seen[dataset] += pending.cpu().numpy()
            pending = torch.zeros(vocab_size, dtype=torch.int64, device=counts.device)
            self._local_seen[dataset] = pending
        pending.add_(counts)

    def synchronize_training_counts(self, distributed=False):
        """Flush rank-local deltas into cumulative CPU counts once per evaluation."""
        if distributed:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("distributed count synchronization requires an initialized process group")
            backend = dist.get_backend()
            default_device = (
                torch.device("cuda", torch.cuda.current_device())
                if backend == "nccl" else torch.device("cpu")
            )
            for dataset, vocab_size in self.vocab_sizes.items():
                if dataset not in self._local_seen:
                    self._local_seen[dataset] = torch.zeros(
                        vocab_size, dtype=torch.int64, device=default_device
                    )

        for dataset in self.vocab_sizes:
            pending = self._local_seen.get(dataset)
            if pending is None:
                continue
            if distributed:
                dist.all_reduce(pending, op=dist.ReduceOp.SUM)
            self.seen[dataset] += pending.cpu().numpy()
            pending.zero_()

    def state_dict(self):
        state = {dataset: counts.copy() for dataset, counts in self.seen.items()}
        # Normal checkpoints follow a synchronized evaluation. This also makes an
        # emergency single-rank checkpoint retain its currently buffered delta.
        for dataset, pending in self._local_seen.items():
            state[dataset] += pending.cpu().numpy()
        return state

    def load_state_dict(self, state):
        for dataset, values in state.items():
            if dataset not in self.seen:
                continue
            values = np.asarray(values, dtype=np.int64)
            if values.shape != self.seen[dataset].shape:
                raise ValueError(f"per-token count shape mismatch for {dataset}")
            self.seen[dataset][...] = values

    def begin_evaluation(self):
        self.pending = {}

    def add_evaluation_batch(self, dataset, split, logits, targets):
        """Aggregate ordinary next-token cross entropy, independent of training loss variants."""
        vocab_size = self.vocab_sizes[dataset]
        flat_logits = logits.detach().reshape(-1, logits.size(-1))
        flat_targets = targets.detach().reshape(-1)
        # Bound the temporary float32 allocation instead of copying all B*T*V logits.
        rows_per_chunk = max(1, 8_000_000 // flat_logits.size(-1))
        losses = torch.cat([
            F.cross_entropy(
                flat_logits[start:start + rows_per_chunk].float(),
                flat_targets[start:start + rows_per_chunk],
                reduction="none",
            ).cpu()
            for start in range(0, flat_logits.size(0), rows_per_chunk)
        ])
        ids = targets.detach().reshape(-1).to("cpu", dtype=torch.long)
        key = (dataset, split)
        if key not in self.pending:
            self.pending[key] = (
                torch.zeros(vocab_size, dtype=torch.float64),
                torch.zeros(vocab_size, dtype=torch.int64),
            )
        sums, counts = self.pending[key]
        sums.scatter_add_(0, ids, losses.to(torch.float64))
        counts += torch.bincount(ids, minlength=vocab_size)[:vocab_size]

    @staticmethod
    def _summary(values):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if not values.size:
            return {key: math.nan for key in ("mean", "median", "std", "skew", "excess_kurtosis", "min", "max", "p10", "p90", "coefficient_of_variation")}
        mean, std = values.mean(), values.std()
        centered = values - mean
        skew = np.mean(centered ** 3) / std ** 3 if std else 0.0
        kurtosis = np.mean(centered ** 4) / std ** 4 - 3 if std else 0.0
        return {
            "mean": mean, "median": np.median(values), "std": std, "skew": skew,
            "excess_kurtosis": kurtosis, "min": values.min(), "max": values.max(),
            "p10": np.percentile(values, 10), "p90": np.percentile(values, 90),
            "coefficient_of_variation": std / mean if mean else math.nan,
        }

    def export(self, iteration):
        self.synchronize_training_counts(distributed=False)
        rows, summaries = [], []
        for dataset, vocab_size in self.vocab_sizes.items():
            split_data = {}
            for split in ("train", "val"):
                sums, counts = self.pending.get(
                    (dataset, split),
                    (torch.zeros(vocab_size), torch.zeros(vocab_size, dtype=torch.long)),
                )
                sums, counts = sums.numpy(), counts.numpy()
                split_data[split] = np.divide(
                    sums, counts, out=np.full(vocab_size, np.nan), where=counts != 0
                )
                split_data[split + "_count"] = counts
            for token_id in range(vocab_size):
                rows.append({
                    "iteration": iteration, "dataset": dataset, "token_id": token_id,
                    "train_loss": float(split_data["train"][token_id]),
                    "train_eval_count": int(split_data["train_count"][token_id]),
                    "val_loss": float(split_data["val"][token_id]),
                    "val_eval_count": int(split_data["val_count"][token_id]),
                    "training_seen_count": int(self.seen[dataset][token_id]),
                })
            for metric, values in (
                ("train_loss", split_data["train"]), ("val_loss", split_data["val"]),
                ("training_seen_count", self.seen[dataset]),
            ):
                summary = self._summary(values)
                populated = (
                    np.count_nonzero(values)
                    if metric == "training_seen_count"
                    else np.isfinite(values).sum()
                )
                summary.update(iteration=iteration, dataset=dataset, metric=metric,
                               populated_tokens=int(populated), vocab_size=vocab_size)
                summaries.append(summary)
        self._append_csv(self.detail_path, rows, self.DETAIL_FIELDS)
        summary_fields = ("iteration", "dataset", "metric", "populated_tokens", "vocab_size",
                          "mean", "median", "std", "skew", "excess_kurtosis", "min", "max",
                          "p10", "p90", "coefficient_of_variation")
        self._append_csv(self.summary_path, summaries, summary_fields)
        self._write_plot(rows, iteration)

    @staticmethod
    def _append_csv(path, rows, fields):
        new_file = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if new_file:
                writer.writeheader()
            writer.writerows(rows)

    def _write_plot(self, latest_rows, iteration):
        """Write a self-contained data page; Plotly itself loads from its official CDN."""
        payload = json.dumps(latest_rows, allow_nan=True)
        html = """<!doctype html><meta charset='utf-8'><title>Per-token metrics</title>
<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>
<h1>Per-token validation loss and training exposure</h1><label>Dataset: <select id='dataset'></select></label>
<div id='plot' style='height:75vh'></div><script>
const rows=PAYLOAD, sel=document.getElementById('dataset');
[...new Set(rows.map(r=>r.dataset))].forEach(x=>sel.add(new Option(x,x)));
function draw(){const d=rows.filter(r=>r.dataset===sel.value && Number.isFinite(r.val_loss)).sort((a,b)=>b.val_loss-a.val_loss);
 const labels=d.map(r=>'token '+r.token_id), hover=d.map(r=>`token=${r.token_id}<br>train loss=${r.train_loss}<br>val samples=${r.val_eval_count}`);
 Plotly.newPlot('plot',[{x:labels,y:d.map(r=>r.val_loss),name:'validation loss',mode:'markers',text:hover,hovertemplate:'%{text}<br>val loss=%{y}<extra></extra>'},
 {x:labels,y:d.map(r=>r.training_seen_count),name:'times seen in training',mode:'markers',yaxis:'y2',visible:'legendonly'}],
 {title:'Evaluation iteration ITERATION (ordered highest to lowest validation loss)',xaxis:{title:'token (validation-loss order)'},yaxis:{title:'validation loss'},yaxis2:{title:'training occurrences',overlaying:'y',side:'right',rangemode:'tozero'},legend:{orientation:'h'}});}
sel.onchange=draw; draw();</script>""".replace("PAYLOAD", payload).replace("ITERATION", str(iteration))
        with open(self.plot_path, "w", encoding="utf-8") as handle:
            handle.write(html)
