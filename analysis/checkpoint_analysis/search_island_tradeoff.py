import argparse
import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class CandidateSpec:
    mode: str  # threshold | target_islands
    value: float
    tag: str


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Search island settings that keep validation loss within a tolerance while "
            "maximizing decode-speed reduction proxy."
        )
    )
    p.add_argument("ckpt_dir", type=str, help="Checkpoint directory containing ckpt.pt")
    p.add_argument(
        "--island_json",
        type=str,
        default=None,
        help="Path to islands_detailed.json (default: <ckpt_dir>/island_analysis/islands_detailed.json)",
    )
    p.add_argument(
        "--thresholds",
        type=str,
        default="0.2,0.3,0.4,0.5",
        help="Comma-separated thresholds to evaluate",
    )
    p.add_argument(
        "--target_islands",
        type=str,
        default="",
        help="Optional comma-separated target island counts per tensor (global target per run)",
    )
    p.add_argument(
        "--provider_mode",
        choices=["top", "mean"],
        default="top",
        help="Representative to use for row replacement inside an island",
    )
    p.add_argument("--eval_dataset", type=str, default=None, help="Dataset name for validation loss")
    p.add_argument("--eval_iters", type=int, default=50, help="Evaluation iterations")
    p.add_argument("--device", type=str, default="cpu", help="Device for eval subprocess")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--loss_tolerance_pct", type=float, default=2.0, help="Max allowed val-loss increase (%)")
    p.add_argument(
        "--pattern",
        type=str,
        default="",
        help="Only mutate tensors containing this substring (simple filter)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: <ckpt_dir>/island_tradeoff_search)",
    )
    return p.parse_args()


def parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def load_ckpt(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt)
    return ckpt, state


def choose_threshold_entry(report: Dict, threshold: float) -> Optional[Dict]:
    best, dist = None, 1e9
    for entry in report.get("threshold_tries", []):
        d = abs(float(entry["threshold"]) - threshold)
        if d < dist:
            best, dist = entry, d
    return best


def choose_target_islands_entry(report: Dict, target: int) -> Optional[Dict]:
    best, dist = None, 10**9
    for entry in report.get("threshold_tries", []):
        num = int(entry.get("summary", {}).get("num_islands", 0))
        d = abs(num - target)
        if d < dist:
            best, dist = entry, d
    return best


def island_rep(weight: torch.Tensor, island: Dict, provider_mode: str) -> torch.Tensor:
    members = island["member_indices"]
    if provider_mode == "mean":
        return weight[members].mean(dim=0)
    providers = island.get("providers", [])
    if providers:
        return weight[int(providers[0]["vector_index"])]
    return weight[members].mean(dim=0)


def build_candidate_ckpt(
    base_ckpt: Dict,
    base_state: Dict,
    reports_by_tensor: Dict[str, Dict],
    spec: CandidateSpec,
    provider_mode: str,
    pattern: str,
) -> Tuple[Dict, Dict]:
    ckpt = {k: v for k, v in base_ckpt.items()}
    state = {k: v.clone() if torch.is_tensor(v) else v for k, v in base_state.items()}
    ckpt["model"] = state

    per_tensor_meta = {}
    total_rows = 0
    routed_rows = 0

    for tensor_name, report in reports_by_tensor.items():
        if pattern and pattern not in tensor_name:
            continue
        if tensor_name not in state:
            continue
        w = state[tensor_name]
        if not torch.is_tensor(w) or not torch.is_floating_point(w) or w.ndim != 2:
            continue

        if spec.mode == "threshold":
            entry = choose_threshold_entry(report, spec.value)
        else:
            entry = choose_target_islands_entry(report, int(spec.value))
        if entry is None:
            continue

        islands = entry.get("islands", [])
        if not islands:
            continue

        total_rows += int(w.shape[0])
        routed_rows += sum(len(island["member_indices"]) for island in islands)

        new_w = w.clone()
        for island in islands:
            rep = island_rep(w, island, provider_mode)
            for idx in island["member_indices"]:
                if 0 <= int(idx) < new_w.shape[0]:
                    new_w[int(idx)] = rep

        state[tensor_name] = new_w
        per_tensor_meta[tensor_name] = {
            "num_islands": len(islands),
            "covered_rows": sum(len(island["member_indices"]) for island in islands),
            "total_rows": int(w.shape[0]),
        }

    proxy = 0.0
    if total_rows > 0:
        proxy = 1.0 - (sum(x["num_islands"] for x in per_tensor_meta.values()) / total_rows)

    return ckpt, {
        "spec": {"mode": spec.mode, "value": spec.value, "tag": spec.tag},
        "tensor_meta": per_tensor_meta,
        "total_rows": total_rows,
        "covered_rows": routed_rows,
        "decode_reduction_proxy": proxy,
    }


def run_eval(candidate_dir: str, eval_dataset: str, eval_iters: int, device: str, dtype: str) -> Dict:
    eval_out = os.path.join(candidate_dir, "eval")
    os.makedirs(eval_out, exist_ok=True)
    cmd = [
        "python3",
        "sample.py",
        "--init_from",
        "resume",
        "--out_dir",
        candidate_dir,
        "--eval_only",
        "--eval_dataset",
        eval_dataset,
        "--eval_iters",
        str(eval_iters),
        "--device",
        device,
        "--dtype",
        dtype,
        "--no-print_model_info",
        "--eval_output_dir",
        eval_out,
    ]
    subprocess.run(cmd, check=True)
    with open(os.path.join(eval_out, "eval_loss.txt"), "r", encoding="utf-8") as f:
        return json.load(f)


def write_plotly(rows: List[Dict], out_html: str):
    import plotly.graph_objects as go

    x = [r["candidate"] for r in rows]
    y_loss = [r["val_loss"] for r in rows]
    y_proxy = [r["decode_reduction_proxy"] for r in rows]
    feasible = [r["feasible"] for r in rows]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y_loss, name="Validation loss"))
    fig.add_trace(go.Bar(x=x, y=y_proxy, name="Decode reduction proxy", yaxis="y2"))
    fig.add_trace(go.Scatter(x=x, y=[1 if f else 0 for f in feasible], mode="markers", name="Feasible (1/0)", yaxis="y2"))
    fig.update_layout(
        title="Island tradeoff search",
        xaxis_title="Candidate",
        yaxis=dict(title="Validation loss"),
        yaxis2=dict(title="Decode reduction proxy", overlaying="y", side="right"),
        barmode="group",
    )
    fig.write_html(out_html, include_plotlyjs="cdn")


def main():
    args = parse_args()
    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
    island_json = args.island_json or os.path.join(args.ckpt_dir, "island_analysis", "islands_detailed.json")
    out_dir = args.out_dir or os.path.join(args.ckpt_dir, "island_tradeoff_search")
    os.makedirs(out_dir, exist_ok=True)

    base_ckpt, base_state = load_ckpt(ckpt_path)
    dataset = args.eval_dataset or base_ckpt.get("config", {}).get("dataset")
    if not dataset:
        raise ValueError("--eval_dataset is required when checkpoint config has no dataset")

    with open(island_json, "r", encoding="utf-8") as f:
        island_payload = json.load(f)
    reports_by_tensor = {r["tensor"]: r for r in island_payload.get("reports", [])}

    specs = [CandidateSpec(mode="threshold", value=v, tag=f"thr_{v}") for v in parse_floats(args.thresholds)]
    specs += [CandidateSpec(mode="target_islands", value=v, tag=f"target_{v}") for v in parse_ints(args.target_islands)]
    if not specs:
        raise ValueError("No candidates provided from --thresholds/--target_islands")

    # baseline evaluation
    baseline_dir = os.path.join(out_dir, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)
    shutil.copy2(ckpt_path, os.path.join(baseline_dir, "ckpt.pt"))
    meta_path = os.path.join(args.ckpt_dir, "meta.pkl")
    if os.path.exists(meta_path):
        shutil.copy2(meta_path, os.path.join(baseline_dir, "meta.pkl"))

    base_eval = run_eval(baseline_dir, dataset, args.eval_iters, args.device, args.dtype)
    base_loss = float(base_eval["val"])
    max_allowed = base_loss * (1 + args.loss_tolerance_pct / 100.0)

    rows = []
    for spec in specs:
        candidate_dir = os.path.join(out_dir, spec.tag)
        os.makedirs(candidate_dir, exist_ok=True)

        cand_ckpt, cand_meta = build_candidate_ckpt(
            base_ckpt=base_ckpt,
            base_state=base_state,
            reports_by_tensor=reports_by_tensor,
            spec=spec,
            provider_mode=args.provider_mode,
            pattern=args.pattern,
        )

        torch.save(cand_ckpt, os.path.join(candidate_dir, "ckpt.pt"))
        if os.path.exists(meta_path):
            shutil.copy2(meta_path, os.path.join(candidate_dir, "meta.pkl"))

        eval_metrics = run_eval(candidate_dir, dataset, args.eval_iters, args.device, args.dtype)
        val_loss = float(eval_metrics["val"])
        feasible = val_loss <= max_allowed

        row = {
            "candidate": spec.tag,
            "mode": spec.mode,
            "value": spec.value,
            "val_loss": val_loss,
            "loss_delta_pct": (val_loss / base_loss - 1) * 100.0,
            "decode_reduction_proxy": cand_meta["decode_reduction_proxy"],
            "total_rows": cand_meta["total_rows"],
            "covered_rows": cand_meta["covered_rows"],
            "feasible": feasible,
        }
        rows.append(row)

        with open(os.path.join(candidate_dir, "candidate_meta.json"), "w", encoding="utf-8") as f:
            json.dump(cand_meta, f, indent=2)

    feasible_rows = [r for r in rows if r["feasible"]]
    best = None
    if feasible_rows:
        best = max(feasible_rows, key=lambda r: (r["decode_reduction_proxy"], -r["val_loss"]))

    summary = {
        "dataset": dataset,
        "baseline_val_loss": base_loss,
        "max_allowed_val_loss": max_allowed,
        "loss_tolerance_pct": args.loss_tolerance_pct,
        "best_candidate": best,
        "rows": rows,
    }

    with open(os.path.join(out_dir, "search_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(out_dir, "search_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "candidate",
            "mode",
            "value",
            "val_loss",
            "loss_delta_pct",
            "decode_reduction_proxy",
            "total_rows",
            "covered_rows",
            "feasible",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if rows:
        try:
            write_plotly(rows, os.path.join(out_dir, "search_dashboard.html"))
        except Exception as exc:
            print(f"Skipped dashboard generation: {exc}")

    print(f"Baseline val loss: {base_loss:.6f}")
    print(f"Max allowed val loss: {max_allowed:.6f}")
    if best:
        print(f"Best candidate: {best['candidate']} (proxy={best['decode_reduction_proxy']:.4f}, val={best['val_loss']:.6f})")
    else:
        print("No feasible candidate met loss tolerance")
    print(f"Wrote {os.path.join(out_dir, 'search_results.json')}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
