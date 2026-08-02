"""Per-token loss/count reporting kept deliberately separate from TensorBoard."""

import csv
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F


class PerTokenMetrics:
    """Accumulate training-token exposure and export evaluation snapshots."""

    DETAIL_FIELDS = (
        "iteration", "dataset", "token_id", "train_loss", "train_eval_count",
        "val_loss", "val_eval_count", "training_seen_count",
    )

    def __init__(self, output_dir, vocab_sizes):
        self.output_dir = output_dir
        self.vocab_sizes = dict(vocab_sizes)
        self.seen = {
            name: np.zeros(size, dtype=np.int64) for name, size in self.vocab_sizes.items()
        }
        self.pending = {}
        os.makedirs(output_dir, exist_ok=True)
        self.detail_path = os.path.join(output_dir, "per_token_metrics.csv")
        self.summary_path = os.path.join(output_dir, "per_token_summary.csv")
        self.plot_path = os.path.join(output_dir, "per_token_metrics.html")

    def count_training_batch(self, dataset, targets):
        values = targets.detach().reshape(-1).to("cpu", dtype=torch.long)
        counts = torch.bincount(values, minlength=self.vocab_sizes[dataset]).numpy()
        self.seen[dataset] += counts[: self.vocab_sizes[dataset]]

    def begin_evaluation(self):
        self.pending = {}

    def add_evaluation_batch(self, dataset, split, logits, targets):
        """Aggregate ordinary next-token cross entropy, independent of training loss variants."""
        vocab_size = self.vocab_sizes[dataset]
        losses = F.cross_entropy(
            logits.detach().float().reshape(-1, logits.size(-1)),
            targets.detach().reshape(-1), reduction="none",
        ).cpu()
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
                summary.update(iteration=iteration, dataset=dataset, metric=metric,
                               populated_tokens=int(np.isfinite(values).sum()), vocab_size=vocab_size)
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
