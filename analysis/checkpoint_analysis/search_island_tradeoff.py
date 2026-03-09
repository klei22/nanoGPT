import argparse
import json
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

import torch
import yaml


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Greedy per-tensor island search: start near 1 island/tensor, then try +1 "
            "island for each tensor individually, selecting the lowest val-loss increase "
            "that remains within tolerance."
        )
    )
    p.add_argument("ckpt_dir", type=str, help="Checkpoint directory containing ckpt.pt")
    p.add_argument(
        "--island_json",
        type=str,
        default=None,
        help="Path to islands_detailed.json (default: <ckpt_dir>/island_analysis/islands_detailed.json)",
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
        "--provider_mode",
        choices=["top", "mean"],
        default="top",
        help="Representative to use for row replacement inside an island",
    )
    p.add_argument("--max_rounds", type=int, default=999, help="Max greedy rounds")
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: <ckpt_dir>/island_tradeoff_search)",
    )
    return p.parse_args()


def load_ckpt(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt)
    return ckpt, state


def island_rep(weight: torch.Tensor, island: Dict, provider_mode: str) -> torch.Tensor:
    members = island["member_indices"]
    if provider_mode == "mean":
        return weight[members].mean(dim=0)
    providers = island.get("providers", [])
    if providers:
        return weight[int(providers[0]["vector_index"])]
    return weight[members].mean(dim=0)


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


def canonical_tensor_candidates(report: Dict) -> List[Dict]:
    """Collapse threshold tries to monotonic candidates by unique num_islands."""
    entries = []
    for e in report.get("threshold_tries", []):
        num = int(e.get("summary", {}).get("num_islands", 0))
        entries.append({
            "num_islands": num,
            "threshold": float(e.get("threshold", 0.0)),
            "islands": e.get("islands", []),
        })
    if not entries:
        return []
    entries.sort(key=lambda x: (x["num_islands"], x["threshold"]))
    out = []
    seen = set()
    for e in entries:
        if e["num_islands"] in seen:
            continue
        seen.add(e["num_islands"])
        out.append(e)
    return out


def initial_candidate_index(cands: List[Dict]) -> int:
    if not cands:
        return 0
    best_i = 0
    best_dist = 10**9
    for i, c in enumerate(cands):
        dist = abs(int(c["num_islands"]) - 1)
        if dist < best_dist:
            best_i, best_dist = i, dist
    return best_i


def build_mutated_state(
    base_state: Dict,
    tensor_candidates: Dict[str, List[Dict]],
    chosen_idx: Dict[str, int],
    provider_mode: str,
) -> Tuple[Dict, Dict]:
    state = {k: v.clone() if torch.is_tensor(v) else v for k, v in base_state.items()}
    meta = {"tensors": {}, "total_rows": 0, "total_islands": 0}

    for tname, idx in chosen_idx.items():
        if tname not in state:
            continue
        w = state[tname]
        if not torch.is_tensor(w) or not torch.is_floating_point(w) or w.ndim != 2:
            continue
        cands = tensor_candidates.get(tname, [])
        if not cands or idx >= len(cands):
            continue

        cand = cands[idx]
        islands = cand.get("islands", [])
        if not islands:
            continue

        new_w = w.clone()
        for island in islands:
            rep = island_rep(w, island, provider_mode)
            for ridx in island.get("member_indices", []):
                ridx = int(ridx)
                if 0 <= ridx < new_w.shape[0]:
                    new_w[ridx] = rep
        state[tname] = new_w

        meta["tensors"][tname] = {
            "num_islands": int(cand["num_islands"]),
            "threshold": float(cand["threshold"]),
            "covered_rows": int(sum(len(i.get("member_indices", [])) for i in islands)),
            "total_rows": int(w.shape[0]),
        }
        meta["total_rows"] += int(w.shape[0])
        meta["total_islands"] += int(cand["num_islands"])

    proxy = 0.0
    if meta["total_rows"] > 0:
        proxy = 1.0 - (meta["total_islands"] / meta["total_rows"])
    meta["decode_reduction_proxy"] = proxy
    return state, meta


def write_yaml(path: str, payload: Dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def to_selected_config(tensor_candidates: Dict[str, List[Dict]], chosen_idx: Dict[str, int]) -> Dict[str, Dict]:
    out = {}
    for tname, idx in chosen_idx.items():
        cands = tensor_candidates.get(tname, [])
        if not cands or idx >= len(cands):
            continue
        c = cands[idx]
        out[tname] = {"idx": idx, "num_islands": int(c["num_islands"]), "threshold": float(c["threshold"])}
    return out


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

    tensor_candidates: Dict[str, List[Dict]] = {}
    for report in island_payload.get("reports", []):
        tname = report.get("tensor")
        if not tname:
            continue
        if args.pattern and args.pattern not in tname:
            continue
        cands = canonical_tensor_candidates(report)
        if cands:
            tensor_candidates[tname] = cands

    if not tensor_candidates:
        raise ValueError("No tensor candidates found. Check --pattern and island analysis JSON.")

    # baseline
    baseline_dir = os.path.join(out_dir, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)
    shutil.copy2(ckpt_path, os.path.join(baseline_dir, "ckpt.pt"))
    meta_path = os.path.join(args.ckpt_dir, "meta.pkl")
    if os.path.exists(meta_path):
        shutil.copy2(meta_path, os.path.join(baseline_dir, "meta.pkl"))

    base_eval = run_eval(baseline_dir, dataset, args.eval_iters, args.device, args.dtype)
    base_loss = float(base_eval["val"])
    max_allowed = base_loss * (1 + args.loss_tolerance_pct / 100.0)

    # initialize around one island per tensor
    chosen_idx = {t: initial_candidate_index(cands) for t, cands in tensor_candidates.items()}
    rounds = []

    current_loss = base_loss
    accepted_rounds = 0

    for rnd in range(1, args.max_rounds + 1):
        tested = []
        best = None

        for tname, idx in chosen_idx.items():
            cands = tensor_candidates.get(tname, [])
            if idx + 1 >= len(cands):
                continue

            trial_idx = dict(chosen_idx)
            trial_idx[tname] = idx + 1
            trial_state, trial_meta = build_mutated_state(base_state, tensor_candidates, trial_idx, args.provider_mode)

            cand_tag = f"round_{rnd}_{tname.replace('.', '_')}"
            cand_dir = os.path.join(out_dir, "candidates", cand_tag)
            os.makedirs(cand_dir, exist_ok=True)
            cand_ckpt = {k: v for k, v in base_ckpt.items()}
            cand_ckpt["model"] = trial_state
            torch.save(cand_ckpt, os.path.join(cand_dir, "ckpt.pt"))
            if os.path.exists(meta_path):
                shutil.copy2(meta_path, os.path.join(cand_dir, "meta.pkl"))

            em = run_eval(cand_dir, dataset, args.eval_iters, args.device, args.dtype)
            vloss = float(em["val"])
            loss_delta_pct = (vloss / base_loss - 1.0) * 100.0
            rec = {
                "tensor": tname,
                "from_idx": idx,
                "to_idx": idx + 1,
                "from_num_islands": int(cands[idx]["num_islands"]),
                "to_num_islands": int(cands[idx + 1]["num_islands"]),
                "val_loss": vloss,
                "loss_delta_pct": loss_delta_pct,
                "decode_reduction_proxy": trial_meta["decode_reduction_proxy"],
                "feasible": vloss <= max_allowed,
            }
            tested.append(rec)

            if best is None or rec["loss_delta_pct"] < best["loss_delta_pct"]:
                best = rec

        round_log = {
            "round": rnd,
            "tested": tested,
            "best_candidate": best,
            "selected": None,
            "stop_reason": None,
            "selected_config": to_selected_config(tensor_candidates, chosen_idx),
            "current_val_loss": current_loss,
        }

        if not tested:
            round_log["stop_reason"] = "no_more_plus_one_candidates"
            rounds.append(round_log)
            break

        if best is None or best["val_loss"] > max_allowed:
            round_log["stop_reason"] = "best_above_loss_threshold"
            rounds.append(round_log)
            break

        # accept best
        chosen_idx[best["tensor"]] = int(best["to_idx"])
        current_loss = float(best["val_loss"])
        accepted_rounds += 1
        round_log["selected"] = best
        round_log["selected_config"] = to_selected_config(tensor_candidates, chosen_idx)
        round_log["current_val_loss"] = current_loss
        rounds.append(round_log)

    # final selected ckpt
    final_state, final_meta = build_mutated_state(base_state, tensor_candidates, chosen_idx, args.provider_mode)
    selected_ckpt = {k: v for k, v in base_ckpt.items()}
    selected_ckpt["model"] = final_state
    selected_dir = os.path.join(out_dir, "selected")
    os.makedirs(selected_dir, exist_ok=True)
    torch.save(selected_ckpt, os.path.join(selected_dir, "ckpt.pt"))
    if os.path.exists(meta_path):
        shutil.copy2(meta_path, os.path.join(selected_dir, "meta.pkl"))

    selected_eval = run_eval(selected_dir, dataset, args.eval_iters, args.device, args.dtype)
    selected_loss = float(selected_eval["val"])

    log_payload = {
        "dataset": dataset,
        "baseline": {
            "val_loss": base_loss,
            "max_allowed_val_loss": max_allowed,
            "loss_tolerance_pct": args.loss_tolerance_pct,
        },
        "search": {
            "strategy": "greedy_plus_one_island_per_tensor",
            "provider_mode": args.provider_mode,
            "pattern": args.pattern,
            "accepted_rounds": accepted_rounds,
            "max_rounds": args.max_rounds,
        },
        "initial_config": {
            t: {"idx": initial_candidate_index(c), "num_islands": int(c[initial_candidate_index(c)]["num_islands"]), "threshold": float(c[initial_candidate_index(c)]["threshold"])}
            for t, c in tensor_candidates.items()
        },
        "rounds": rounds,
        "selected": {
            "config": to_selected_config(tensor_candidates, chosen_idx),
            "val_loss": selected_loss,
            "loss_delta_pct": (selected_loss / base_loss - 1.0) * 100.0,
            "decode_reduction_proxy": final_meta["decode_reduction_proxy"],
            "tensor_meta": final_meta["tensors"],
        },
    }

    yaml_path = os.path.join(out_dir, "search_log.yaml")
    write_yaml(yaml_path, log_payload)

    with open(os.path.join(out_dir, "search_results.json"), "w", encoding="utf-8") as f:
        json.dump(log_payload, f, indent=2)

    print(f"Baseline val loss: {base_loss:.6f}")
    print(f"Max allowed val loss: {max_allowed:.6f}")
    print(f"Selected val loss: {selected_loss:.6f}")
    print(f"Accepted rounds: {accepted_rounds}")
    print(f"Wrote {yaml_path}")
    print(f"Wrote {os.path.join(out_dir, 'search_results.json')}")


if __name__ == "__main__":
    main()
