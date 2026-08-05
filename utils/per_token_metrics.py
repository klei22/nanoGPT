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
        "iteration", "dataset", "token_id", "token_text_escaped", "train_loss", "train_eval_count",
        "val_loss", "val_eval_count", "training_seen_count",
    )

    def __init__(self, output_dir, vocab_sizes, token_texts=None):
        self.output_dir = output_dir
        self.vocab_sizes = dict(vocab_sizes)
        self.token_texts = token_texts or {}
        self.seen = {
            name: np.zeros(size, dtype=np.int64) for name, size in self.vocab_sizes.items()
        }
        self.pending = {}
        os.makedirs(output_dir, exist_ok=True)
        self.detail_path = os.path.join(output_dir, "per_token_metrics.csv")
        self.summary_path = os.path.join(output_dir, "per_token_summary.csv")
        self.plot_path = os.path.join(output_dir, "per_token_metrics.html")
        self._ensure_detail_schema()

    def _ensure_detail_schema(self):
        """Upgrade detail CSVs written before escaped token text was added."""
        if not os.path.exists(self.detail_path) or os.path.getsize(self.detail_path) == 0:
            return
        with open(self.detail_path, newline="", encoding="utf-8") as handle:
            raw_rows = list(csv.reader(handle))
        if not raw_rows or tuple(raw_rows[0]) == self.DETAIL_FIELDS:
            return

        legacy_fields = tuple(field for field in self.DETAIL_FIELDS
                              if field != "token_text_escaped")
        if tuple(raw_rows[0]) != legacy_fields:
            raise ValueError(
                f"Unsupported per-token metrics CSV schema in {self.detail_path}: "
                f"{raw_rows[0]}"
            )

        migrated = []
        for values in raw_rows[1:]:
            # A prior interrupted run may already have appended canonical rows
            # beneath the legacy header. Recover both row shapes.
            fields = self.DETAIL_FIELDS if len(values) >= len(self.DETAIL_FIELDS) else legacy_fields
            row = dict(zip(fields, values))
            dataset = row.get("dataset", "")
            token_id = int(row["token_id"])
            row["token_text_escaped"] = (
                row.get("token_text_escaped")
                or self.token_texts.get(dataset, {}).get(token_id, "")
            )
            migrated.append(row)

        temporary_path = self.detail_path + ".tmp"
        with open(temporary_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.DETAIL_FIELDS)
            writer.writeheader()
            writer.writerows(migrated)
        os.replace(temporary_path, self.detail_path)

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
                    "token_text_escaped": self.token_texts.get(dataset, {}).get(token_id, ""),
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
        self._write_plot(self._read_detail_rows(), summaries, iteration)

    @staticmethod
    def _append_csv(path, rows, fields):
        new_file = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if new_file:
                writer.writeheader()
            writer.writerows(rows)

    def _read_detail_rows(self):
        """Load all snapshots so the HTML can plot a token's history."""
        rows = []
        with open(self.detail_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({
                    "iteration": int(row["iteration"]),
                    "dataset": row["dataset"],
                    "token_id": int(row["token_id"]),
                    "token_text_escaped": row.get("token_text_escaped", ""),
                    "train_loss": float(row["train_loss"]),
                    "train_eval_count": int(row["train_eval_count"]),
                    "val_loss": float(row["val_loss"]),
                    "val_eval_count": int(row["val_eval_count"]),
                    "training_seen_count": int(row["training_seen_count"]),
                })
        return rows

    def _write_plot(self, latest_rows, summaries, iteration):
        """Write a self-contained data page; Plotly itself loads from its official CDN."""
        payload = json.dumps(latest_rows, allow_nan=True)
        summary_payload = json.dumps(summaries, allow_nan=True)
        html = """<!doctype html><meta charset='utf-8'><title>Per-token metrics</title>
<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>
<h1>Per-token validation loss and training exposure</h1><label>Dataset: <select id='dataset'></select></label>
<div id='plotlyStatus' style='color:#b00020;font-weight:bold'></div>
<h2>Summary statistics</h2><div id='summary'></div>
<div id='lossPlot' style='height:75vh'></div>
<div id='trainingLossPlot' style='height:75vh'></div>
<div id='occurrencePlot' style='height:75vh'></div>
<h2>Token history</h2>
<p>Select one or more tokens (Ctrl/Cmd-click to toggle individual entries):</p>
<select id='tokens' multiple size='12' style='min-width:24em'></select>
<div id='iterationPlot' style='height:65vh'></div>
<div id='appearancePlot' style='height:65vh'></div><script>
const rows=PAYLOAD, summaries=SUMMARY_PAYLOAD, sel=document.getElementById('dataset');
const tokenSel=document.getElementById('tokens');
function renderPlot(plotId,traces,layout){try{return Plotly.newPlot(plotId,traces,layout).catch(error=>showPlotError(plotId,error));}catch(error){showPlotError(plotId,error);}}
function showPlotError(plotId,error){const target=document.getElementById(plotId);target.innerHTML=`<p style="color:#b00020"><strong>Unable to render ${plotId}:</strong> ${String(error)}</p>`;console.error(`Unable to render ${plotId}`,error);}
function addScaleControls(plotId,hasRight){const box=document.createElement('div');box.innerHTML=`<strong>Y-axis scale:</strong> <label><input id="${plotId}LeftLog" type="checkbox"> left logarithmic</label>${hasRight?` &nbsp; <label><input id="${plotId}RightLog" type="checkbox"> right logarithmic</label>`:''}`;document.getElementById(plotId).before(box);
 box.querySelectorAll('input').forEach(input=>input.onchange=()=>Plotly.relayout(plotId,input.id.endsWith('RightLog')?{'yaxis2.type':input.checked?'log':'linear'}:{'yaxis.type':input.checked?'log':'linear'}));}
function axisType(plotId,side='Left'){return document.getElementById(`${plotId}${side}Log`).checked?'log':'linear';}
addScaleControls('lossPlot',true);addScaleControls('trainingLossPlot',true);addScaleControls('occurrencePlot',true);addScaleControls('iterationPlot',true);addScaleControls('appearancePlot',false);
[...new Set(rows.map(r=>r.dataset))].forEach(x=>sel.add(new Option(x,x)));
function label(r){return `token ${r.token_id} '${r.token_text_escaped}'`;}
function hover(r){return `token=${r.token_id}<br>text='${r.token_text_escaped}'<br>train loss=${r.train_loss}<br>val samples=${r.val_eval_count}<br>times seen=${r.training_seen_count}`;}
function latestRows(){const datasetRows=rows.filter(r=>r.dataset===sel.value), latestIteration=Math.max(...datasetRows.map(r=>r.iteration)); return datasetRows.filter(r=>r.iteration===latestIteration);}
function draw(){const available=latestRows().filter(r=>Number.isFinite(r.val_loss));
 const byLoss=[...available].sort((a,b)=>b.val_loss-a.val_loss), lossLabels=byLoss.map(label), lossHover=byLoss.map(hover);
 renderPlot('lossPlot',[{x:lossLabels,y:byLoss.map(r=>r.val_loss),name:'validation loss',mode:'markers',text:lossHover,hovertemplate:'%{text}<br>val loss=%{y}<extra></extra>'},
 {x:lossLabels,y:byLoss.map(r=>r.training_seen_count),name:'times seen in training',mode:'markers',yaxis:'y2'}],
 {title:'Evaluation iteration ITERATION (ordered highest to lowest validation loss)',xaxis:{title:'token (validation-loss order)'},yaxis:{title:'validation loss',type:axisType('lossPlot')},yaxis2:{title:'training occurrences',type:axisType('lossPlot','Right'),overlaying:'y',side:'right',rangemode:'tozero'},legend:{orientation:'h'}});}
function drawTrainingLoss(){const available=latestRows().filter(r=>Number.isFinite(r.train_loss)), d=[...available].sort((a,b)=>b.train_loss-a.train_loss), labels=d.map(label), h=d.map(hover);
 renderPlot('trainingLossPlot',[{x:labels,y:d.map(r=>r.train_loss),name:'sampled training loss',mode:'markers',text:h,hovertemplate:'%{text}<br>train loss=%{y}<extra></extra>'},
 {x:labels,y:d.map(r=>r.training_seen_count),name:'times seen in training',mode:'markers',yaxis:'y2'}],
 {title:'Evaluation iteration ITERATION (ordered highest to lowest sampled training loss)',xaxis:{title:'token (training-loss order)'},yaxis:{title:'sampled training loss',type:axisType('trainingLossPlot')},yaxis2:{title:'training occurrences',type:axisType('trainingLossPlot','Right'),overlaying:'y',side:'right',rangemode:'tozero'},legend:{orientation:'h'}});}
function drawOccurrence(){const d=latestRows().sort((a,b)=>a.training_seen_count-b.training_seen_count || a.token_id-b.token_id), labels=d.map(label), h=d.map(hover);
 renderPlot('occurrencePlot',[{x:labels,y:d.map(r=>r.training_seen_count),name:'times seen in training',mode:'markers',text:h,hovertemplate:'%{text}<extra></extra>'},
 {x:labels,y:d.map(r=>r.val_loss),name:'validation loss',mode:'markers',yaxis:'y2',text:h,hovertemplate:'%{text}<br>val loss=%{y}<extra></extra>'},
 {x:labels,y:d.map(r=>r.train_loss),name:'sampled training loss',mode:'markers',yaxis:'y2',text:h,hovertemplate:'%{text}<br>train loss=%{y}<extra></extra>'}],
 {title:'Evaluation iteration ITERATION (ordered lowest to highest training occurrence)',xaxis:{title:'token (training-occurrence order)'},yaxis:{title:'training occurrences',type:axisType('occurrencePlot'),rangemode:'tozero'},yaxis2:{title:'loss',type:axisType('occurrencePlot','Right'),overlaying:'y',side:'right'},legend:{orientation:'h'}});}
function drawSummary(){const d=summaries.filter(r=>r.dataset===sel.value), fields=['populated_tokens','vocab_size','mean','median','std','skew','excess_kurtosis','min','max','p10','p90','coefficient_of_variation'];
 let table='<table border="1" cellpadding="5" style="border-collapse:collapse"><thead><tr><th>metric</th>'+fields.map(x=>`<th>${x}</th>`).join('')+'</tr></thead><tbody>';
 table+=d.map(r=>'<tr><th>'+r.metric+'</th>'+fields.map(f=>`<td>${typeof r[f]==='number' ? Number(r[f]).toPrecision(6) : r[f]}</td>`).join('')+'</tr>').join('')+'</tbody></table>'; document.getElementById('summary').innerHTML=table;}
function populateTokens(){tokenSel.replaceChildren(); const d=latestRows(), worst=[...d].filter(r=>Number.isFinite(r.val_loss)).sort((a,b)=>b.val_loss-a.val_loss).slice(0,5).map(r=>r.token_id);
 d.sort((a,b)=>a.token_id-b.token_id).forEach(r=>{const o=new Option(label(r),r.token_id);o.selected=worst.includes(r.token_id);tokenSel.add(o);});}
function selectedTokens(){return [...tokenSel.selectedOptions].map(o=>Number(o.value));}
function historyTraces(xField){const palette=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']; const traces=[];
 selectedTokens().forEach((id,i)=>{const d=rows.filter(r=>r.dataset===sel.value&&r.token_id===id).sort((a,b)=>a.iteration-b.iteration), name=d.length?label(d[0]):`token ${id}`, color=palette[i%palette.length];
  traces.push({x:d.map(r=>r[xField]),y:d.map(r=>r.val_loss),name:`${name} validation`,mode:'lines+markers',legendgroup:String(id),line:{color}});
  traces.push({x:d.map(r=>r[xField]),y:d.map(r=>r.train_loss),name:`${name} train`,mode:'lines+markers',legendgroup:String(id),line:{color,dash:'dot'}});}); return traces;}
function appearanceByIterationTraces(){const palette=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];return selectedTokens().map((id,i)=>{const d=rows.filter(r=>r.dataset===sel.value&&r.token_id===id).sort((a,b)=>a.iteration-b.iteration),name=d.length?label(d[0]):`token ${id}`;return {x:d.map(r=>r.iteration),y:d.map(r=>r.training_seen_count),name:`${name} cumulative appearances`,mode:'lines+markers',legendgroup:String(id),yaxis:'y2',line:{color:palette[i%palette.length],dash:'dash'}};});}
function drawHistory(){renderPlot('iterationPlot',[...historyTraces('iteration'),...appearanceByIterationTraces()],{title:'Selected-token loss and cumulative appearances vs iteration',xaxis:{title:'training iteration'},yaxis:{title:'cross-entropy loss',type:axisType('iterationPlot')},yaxis2:{title:'cumulative training appearances',type:axisType('iterationPlot','Right'),overlaying:'y',side:'right',rangemode:'tozero'},legend:{orientation:'h'}});
 renderPlot('appearancePlot',historyTraces('training_seen_count'),{title:'Selected-token loss vs cumulative appearances',xaxis:{title:'cumulative training appearances'},yaxis:{title:'cross-entropy loss',type:axisType('appearancePlot')},legend:{orientation:'h'}});}
function drawAll(){draw();drawTrainingLoss();drawOccurrence();drawSummary();populateTokens();drawHistory();}
function initialize(){drawSummary();populateTokens();if(typeof Plotly!=='undefined'){drawAll();return;}const status=document.getElementById('plotlyStatus');status.textContent='Primary Plotly CDN unavailable; loading fallback…';const fallback=document.createElement('script');fallback.src='https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js';fallback.onload=()=>{status.textContent='';drawAll();};fallback.onerror=()=>{status.textContent='Plotly could not be loaded. Check network access to cdn.plot.ly or cdn.jsdelivr.net, then reload this file.';};document.head.appendChild(fallback);}
sel.onchange=drawAll;tokenSel.onchange=drawHistory;initialize();</script>""".replace("SUMMARY_PAYLOAD", summary_payload).replace("PAYLOAD", payload).replace("ITERATION", str(iteration))
        with open(self.plot_path, "w", encoding="utf-8") as handle:
            handle.write(html)
