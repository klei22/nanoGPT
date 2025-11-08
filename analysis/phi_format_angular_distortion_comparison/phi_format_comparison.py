#!/usr/bin/env python3
"""
phi_angle_eval_sweep.py

Enhancements:
- Sweep over multiple constants for "phi numbers" via --phi-consts (comma-separated list).
  Allowed constants: golden_ratio, sqrt2 (alias: phi), e, pi, 1, 2, 3, 4, 5.
  Each constant is computed in fp16 then promoted to fp32.
- Optional log-scale for Y axis (--logy).
- Optional "restricted-phi" mode (--restricted-phi) where the codebook excludes 0 so the space is strictly symmetric.
  If activated, the restricted-phi series is added to the plot and CSV alongside standard phi.

Other features (kept):
- CUDA acceleration (default on) using PyTorch when available; otherwise NumPy/CPU.
- Bit sweeps over even totals for both phi and int.
- Trials per bit setting; mean ± std of ANGLE (degrees) plotted as error bars.
"""

from __future__ import annotations
import argparse
import math
import numpy as np

# Optional torch import guarded
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def twos_comp_range(bits: int) -> tuple[int, int]:
    if bits <= 0:
        raise ValueError("bits must be positive")
    mn = -(1 << (bits - 1))
    mx = (1 << (bits - 1)) - 1
    return mn, mx


def build_phi_codebook_np(a_bits: int, b_bits: int, const_val: np.float32, restricted: bool) -> np.ndarray:
    amin, amax = twos_comp_range(a_bits)
    bmin, bmax = twos_comp_range(b_bits)
    a_vals = np.arange(amin, amax + 1, dtype=np.int32)
    b_vals = np.arange(bmin, bmax + 1, dtype=np.int32)
    codebook = a_vals[:, None].astype(np.float32) + (np.float32(const_val) * b_vals[None, :].astype(np.float32))
    codebook = np.sort(codebook.ravel())
    if restricted:
        codebook = codebook[codebook != 0.0]
    return codebook


def nearest_in_codebook_np(u: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(codebook, u)
    idx = np.clip(idx, 1, len(codebook) - 1)
    prev_vals = codebook[idx - 1]
    next_vals = codebook[idx]
    choose_prev = (np.abs(u - prev_vals) <= np.abs(next_vals - u))
    return np.where(choose_prev, prev_vals, next_vals)


def int_quantize_codes_np(v_over_s: np.ndarray, bits: int) -> np.ndarray:
    qmin, qmax = twos_comp_range(bits)
    codes = np.clip(np.rint(v_over_s), qmin, qmax).astype(np.int32)
    return codes.astype(np.float32)


def cosine_similarity_np(v: np.ndarray, q: np.ndarray) -> float:
    v64 = v.astype(np.float64, copy=False)
    q64 = q.astype(np.float64, copy=False)
    num = float(np.dot(v64, q64))
    den = float(np.linalg.norm(v64) * np.linalg.norm(q64))
    if den == 0.0:
        return 0.0
    return num / den


def s0_for_int_np(v: np.ndarray, bits: int) -> float:
    _, qmax = twos_comp_range(bits)
    vmax = float(np.quantile(np.abs(v), 0.999))
    vmax = max(vmax, 1e-12)
    return vmax / max(1, qmax)


def s0_for_phi_np(v: np.ndarray, a_bits: int, b_bits: int, const_val: np.float32) -> float:
    amin, amax = twos_comp_range(a_bits)
    bmin, bmax = twos_comp_range(b_bits)
    aabs = max(abs(amin), abs(amax))
    babs = max(abs(bmin), abs(bmax))
    max_code_abs = float(aabs + float(const_val) * babs)
    vmax = float(np.quantile(np.abs(v), 0.999))
    vmax = max(vmax, 1e-12)
    return vmax / max(max_code_abs, 1e-12)


# Torch versions (if available)
def build_phi_codebook_torch(a_bits: int, b_bits: int, const_val: float, restricted: bool, device: 'torch.device') -> 'torch.Tensor':
    amin, amax = twos_comp_range(a_bits)
    bmin, bmax = twos_comp_range(b_bits)
    a_vals = torch.arange(amin, amax + 1, dtype=torch.float32, device=device)
    b_vals = torch.arange(bmin, bmax + 1, dtype=torch.float32, device=device)
    codebook = a_vals[:, None] + torch.tensor(const_val, dtype=torch.float32, device=device) * b_vals[None, :]
    codebook = codebook.reshape(-1).sort().values
    if restricted:
        codebook = codebook[codebook != 0.0]
    return codebook


def nearest_in_codebook_torch(u: 'torch.Tensor', codebook: 'torch.Tensor') -> 'torch.Tensor':
    idx = torch.searchsorted(codebook, u)
    idx = torch.clamp(idx, 1, codebook.numel() - 1)
    prev_vals = codebook[idx - 1]
    next_vals = codebook[idx]
    choose_prev = (torch.abs(u - prev_vals) <= torch.abs(next_vals - u))
    return torch.where(choose_prev, prev_vals, next_vals)


def int_quantize_codes_torch(v_over_s: 'torch.Tensor', bits: int) -> 'torch.Tensor':
    qmin, qmax = twos_comp_range(bits)
    codes = torch.clamp(torch.round(v_over_s), qmin, qmax)
    return codes.to(torch.float32)


def cosine_similarity_torch(v: 'torch.Tensor', q: 'torch.Tensor') -> float:
    v64 = v.double()
    q64 = q.double()
    num = torch.dot(v64, q64).item()
    den = (torch.linalg.norm(v64) * torch.linalg.norm(q64)).item()
    if den == 0.0:
        return 0.0
    return num / den


def s0_for_int_torch(v: 'torch.Tensor', bits: int) -> float:
    _, qmax = twos_comp_range(bits)
    vmax = torch.quantile(torch.abs(v), torch.tensor(0.999, dtype=v.dtype, device=v.device)).item()
    vmax = max(vmax, 1e-12)
    return vmax / max(1, qmax)


def s0_for_phi_torch(v: 'torch.Tensor', a_bits: int, b_bits: int, const_val: float) -> float:
    amin, amax = twos_comp_range(a_bits)
    bmin, bmax = twos_comp_range(b_bits)
    aabs = max(abs(amin), abs(amax))
    babs = max(abs(bmin), abs(bmax))
    max_code_abs = float(aabs + float(const_val) * babs)
    vmax = torch.quantile(torch.abs(v), torch.tensor(0.999, dtype=v.dtype, device=v.device)).item()
    vmax = max(vmax, 1e-12)
    return vmax / max(max_code_abs, 1e-12)


def optimize_scale_np(v: np.ndarray, quantize_fn, s0: float, grid_steps: int, grid_span: float, refine_steps: int, refine_span: float) -> tuple[float, float]:
    best_s, best_cos = None, -1.0
    s_values1 = s0 * (2.0 ** np.linspace(-grid_span, grid_span, grid_steps))
    for s in s_values1:
        if s <= 0: continue
        q = quantize_fn(v, float(s))
        c = cosine_similarity_np(v, q)
        if c > best_cos:
            best_cos, best_s = c, float(s)
    s_values2 = best_s * (2.0 ** np.linspace(-refine_span, refine_span, refine_steps))
    for s in s_values2:
        if s <= 0: continue
        q = quantize_fn(v, float(s))
        c = cosine_similarity_np(v, q)
        if c > best_cos:
            best_cos, best_s = c, float(s)
    return best_s, best_cos


def optimize_scale_torch(v: 'torch.Tensor', quantize_fn, s0: float, grid_steps: int, grid_span: float, refine_steps: int, refine_span: float) -> tuple[float, float]:
    best_s, best_cos = None, -1.0
    s_values1 = s0 * (2.0 ** torch.linspace(-grid_span, grid_span, steps=grid_steps, device=v.device, dtype=torch.float32))
    for s in s_values1:
        if s.item() <= 0: continue
        q = quantize_fn(v, float(s.item()))
        c = cosine_similarity_torch(v, q)
        if c > best_cos:
            best_cos, best_s = c, float(s.item())
    s_values2 = best_s * (2.0 ** torch.linspace(-refine_span, refine_span, steps=refine_steps, device=v.device, dtype=torch.float32))
    for s in s_values2:
        if s.item() <= 0: continue
        q = quantize_fn(v, float(s.item()))
        c = cosine_similarity_torch(v, q)
        if c > best_cos:
            best_cos, best_s = c, float(s.item())
    return best_s, best_cos


def pick_constant_fp16_to_fp32(name: str) -> tuple[str, float, float]:
    nm = name.lower().strip()
    if nm == "golden_ratio":
        base = (1.0 + np.sqrt(5.0)) / 2.0
    elif nm in ("sqrt2", "phi"):
        base = np.sqrt(2.0)
        nm = "sqrt2"
    elif nm == "e":
        base = math.e
    elif nm == "pi":
        base = math.pi
    elif nm in {"1","2","3","4","5"}:
        base = float(nm)
    else:
        raise ValueError(f"Unknown constant '{name}'. Choose from golden_ratio, sqrt2, e, pi, 1, 2, 3, 4, 5.")
    fp16_val = np.float16(base)
    fp32_promoted = float(np.float32(fp16_val))
    return nm, float(fp16_val), fp32_promoted


def run_trials_for_bits_one_const(
    dim: int,
    mean: float,
    std: float,
    trials: int,
    phi_bits: int,
    int_bits: int,
    grid_steps: int,
    grid_span: float,
    refine_steps: int,
    refine_span: float,
    use_cuda: bool,
    seed: int | None,
    const_name: str,
    restricted: bool,
):
    # constant selection (fp16 -> fp32 promoted)
    cname, c_fp16, c_value = pick_constant_fp16_to_fp32(const_name)

    if phi_bits % 2 != 0:
        raise ValueError("phi-bits must be even (split evenly between a and b).")
    a_bits = b_bits = phi_bits // 2

    # Choose backend
    use_torch = use_cuda and TORCH_AVAILABLE
    device = None
    if use_torch:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_torch = (device.type == "cuda")
    if use_cuda and not use_torch:
        print("[info] CUDA requested but not available; falling back to NumPy/CPU.]")

    phi_cos_vals, int_cos_vals, rphi_cos_vals = [], [], []

    # Pre-build codebooks (backend-specific)
    if use_torch:
        codebook_phi = build_phi_codebook_torch(a_bits, b_bits, c_value, restricted=False, device=device)
        codebook_rphi = build_phi_codebook_torch(a_bits, b_bits, c_value, restricted=True, device=device) if restricted else None
    else:
        codebook_phi = build_phi_codebook_np(a_bits, b_bits, c_value, restricted=False)
        codebook_rphi = build_phi_codebook_np(a_bits, b_bits, c_value, restricted=True) if restricted else None

    # Trial loop
    for t in range(trials):
        if seed is not None:
            rng_seed = seed + t
            np.random.seed(rng_seed)
            if use_torch:
                torch.manual_seed(rng_seed)

        if use_torch:
            v = torch.tensor(np.random.normal(loc=mean, scale=std, size=(dim,)), dtype=torch.float32, device=device)

            def phi_quantizer(v_in: 'torch.Tensor', s: float) -> 'torch.Tensor':
                u = v_in / torch.tensor(s, dtype=torch.float32, device=v_in.device)
                return nearest_in_codebook_torch(u, codebook_phi)

            def rphi_quantizer(v_in: 'torch.Tensor', s: float) -> 'torch.Tensor':
                u = v_in / torch.tensor(s, dtype=torch.float32, device=v_in.device)
                return nearest_in_codebook_torch(u, codebook_rphi)

            def int_quantizer(v_in: 'torch.Tensor', s: float) -> 'torch.Tensor':
                u = v_in / torch.tensor(s, dtype=torch.float32, device=v_in.device)
                return int_quantize_codes_torch(u, int_bits)

            s0_phi = s0_for_phi_torch(v, a_bits, b_bits, c_value)
            s0_int = s0_for_int_torch(v, int_bits)

            _, best_cos_phi = optimize_scale_torch(v, phi_quantizer, s0_phi, grid_steps, grid_span, refine_steps, refine_span)
            _, best_cos_int = optimize_scale_torch(v, int_quantizer, s0_int, grid_steps, grid_span, refine_steps, refine_span)
            phi_cos_vals.append(best_cos_phi)
            int_cos_vals.append(best_cos_int)

            if restricted:
                s0_rphi = s0_phi  # heuristic reuse
                _, best_cos_rphi = optimize_scale_torch(v, rphi_quantizer, s0_rphi, grid_steps, grid_span, refine_steps, refine_span)
                rphi_cos_vals.append(best_cos_rphi)

        else:
            v = np.random.normal(loc=mean, scale=std, size=(dim,)).astype(np.float32)

            def phi_quantizer(v_in: np.ndarray, s: float) -> np.ndarray:
                u = v_in / np.float32(s)
                return nearest_in_codebook_np(u, codebook_phi).astype(np.float32)

            def rphi_quantizer(v_in: np.ndarray, s: float) -> np.ndarray:
                u = v_in / np.float32(s)
                return nearest_in_codebook_np(u, codebook_rphi).astype(np.float32)

            def int_quantizer(v_in: np.ndarray, s: float) -> np.ndarray:
                u = v_in / np.float32(s)
                return int_quantize_codes_np(u, int_bits)

            s0_phi = s0_for_phi_np(v, a_bits, b_bits, c_value)
            s0_int = s0_for_int_np(v, int_bits)

            _, best_cos_phi = optimize_scale_np(v, phi_quantizer, s0_phi, grid_steps, grid_span, refine_steps, refine_span)
            _, best_cos_int = optimize_scale_np(v, int_quantizer, s0_int, grid_steps, grid_span, refine_steps, refine_span)
            phi_cos_vals.append(best_cos_phi)
            int_cos_vals.append(best_cos_int)

            if restricted:
                s0_rphi = s0_phi
                _, best_cos_rphi = optimize_scale_np(v, rphi_quantizer, s0_rphi, grid_steps, grid_span, refine_steps, refine_span)
                rphi_cos_vals.append(best_cos_rphi)

    # convert to angles
    def cos_to_deg(cos_list):
        return [math.degrees(math.acos(max(-1.0, min(1.0, c)))) for c in cos_list]

    phi_angles = cos_to_deg(phi_cos_vals)
    int_angles = cos_to_deg(int_cos_vals)
    rphi_angles = cos_to_deg(rphi_cos_vals) if restricted else []

    # stats
    def mean_std(arr):
        arr = np.array(arr, dtype=np.float64)
        if arr.size == 0:
            return 0.0, 0.0
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        return mean, std

    phi_mean, phi_std = mean_std(phi_angles)
    int_mean, int_std = mean_std(int_angles)
    rphi_mean, rphi_std = (mean_std(rphi_angles) if restricted else (float('nan'), float('nan')))

    return {
        "phi_bits": phi_bits,
        "int_bits": int_bits,
        "phi_angle_mean": phi_mean,
        "phi_angle_std": phi_std,
        "int_angle_mean": int_mean,
        "int_angle_std": int_std,
        "rphi_angle_mean": rphi_mean,
        "rphi_angle_std": rphi_std,
        "const_name": cname,
        "const_fp32": c_value,
        "bits": phi_bits,  # convenience
    }


def parse_const_list(const_list: str) -> list[str]:
    raw = [c.strip() for c in const_list.split(",") if c.strip()]
    if not raw:
        return ["sqrt2"]
    return raw


def main():
    parser = argparse.ArgumentParser(description="Phi/int angle-preserving quantization with CUDA, bit & constant sweeps, and restricted-phi option.")
    parser.add_argument("--dim", type=int, default=384, help="Vector dimensionality (default: 384).")
    parser.add_argument("--mean", type=float, default=0.0, help="Gaussian mean (default: 0.0).")
    parser.add_argument("--std", type=float, default=0.02, help="Gaussian stddev (default: 0.02).")
    parser.add_argument("--trials", type=int, default=20, help="Number of independent random vectors per bit setting (default: 20).")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed (default: None).")

    # Bit sweeps (even numbers, inclusive)
    parser.add_argument("--phi-bits-start", type=int, default=4, help="Start total bits for phi format (even).")
    parser.add_argument("--phi-bits-end", type=int, default=12, help="End total bits for phi format (even).")
    parser.add_argument("--int-bits-start", type=int, default=4, help="Start bits for int format (even).")
    parser.add_argument("--int-bits-end", type=int, default=12, help="End bits for int format (even).")

    # Scale search
    parser.add_argument("--grid-steps", type=int, default=129, help="Coarse grid steps around s0 in log2 (default: 129).")
    parser.add_argument("--grid-span", type=float, default=8.0, help="Coarse log2 span around s0 (default: 8.0).")
    parser.add_argument("--refine-steps", type=int, default=129, help="Refinement grid steps (default: 129).")
    parser.add_argument("--refine-span", type=float, default=1.0, help="Refinement log2 span around best s (default: 1.0).")

    # Constant sweep
    parser.add_argument("--phi-consts", type=str, default="sqrt2",
                        help="Comma-separated constants to sweep: golden_ratio,sqrt2,e,pi,1,2,3,4,5 (default: sqrt2).")

    # Restricted-phi toggle
    parser.add_argument("--restricted-phi", action="store_true", help="Also evaluate restricted-phi (no zero representable).")

    # CUDA toggle
    parser.add_argument("--cuda", dest="cuda", action="store_true", help="Enable CUDA acceleration if available (default).")
    parser.add_argument("--no-cuda", dest="cuda", action="store_false", help="Disable CUDA acceleration.")
    parser.set_defaults(cuda=True)

    # Plot options
    parser.add_argument("--logy", action="store_true", help="Use logarithmic y-axis for angle plot.")

    # Output
    parser.add_argument("--out-png", type=str, default="angle_vs_bits.png", help="Where to save the plot PNG.")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV path to dump summary (long-form).")

    args = parser.parse_args()

    consts = parse_const_list(args.phi_consts)

    # Informational printout
    print("=== Setup ===")
    print(f"dim: {args.dim}, trials per setting: {args.trials}, distribution: N({args.mean}, {args.std}^2)")
    print(f"phi constants: {consts}")
    print(f"phi bits sweep: {args.phi_bits_start}..{args.phi_bits_end} (even) | int bits sweep: {args.int_bits_start}..{args.int_bits_end} (even)")
    print(f"grid steps/span: {args.grid_steps} / ±{args.grid_span} (log2) | refine steps/span: {args.refine_steps} / ±{args.refine_span} (log2)")
    print(f"restricted-phi: {args.restricted_phi} | logy: {args.logy}")
    print(f"cuda requested: {args.cuda} | torch available: {TORCH_AVAILABLE}")

    # Build sweeps (even numbers)
    def even_range(a,b):
        start = a + (a % 2)
        end = b - (b % 2) + (0 if b % 2 == 0 else 0)
        return list(range(start, end + 1, 2))

    phi_bits_list = even_range(args.phi_bits_start, args.phi_bits_end)
    int_bits_list = even_range(args.int_bits_start, args.int_bits_end)
    all_bits = sorted(set(phi_bits_list) | set(int_bits_list))

    # Collect results in long-form records
    import pandas as pd
    records = []

    for cname in consts:
        for bits in all_bits:
            phi_bits = bits if bits in phi_bits_list else None
            int_bits = bits if bits in int_bits_list else None

            if phi_bits is not None and int_bits is not None:
                res = run_trials_for_bits_one_const(
                    dim=args.dim, mean=args.mean, std=args.std, trials=args.trials,
                    phi_bits=phi_bits, int_bits=int_bits,
                    grid_steps=args.grid_steps, grid_span=args.grid_span,
                    refine_steps=args.refine_steps, refine_span=args.refine_span,
                    use_cuda=args.cuda, seed=args.seed,
                    const_name=cname, restricted=args.restricted_phi
                )
            elif phi_bits is not None:
                res = run_trials_for_bits_one_const(
                    dim=args.dim, mean=args.mean, std=args.std, trials=args.trials,
                    phi_bits=phi_bits, int_bits=8,
                    grid_steps=args.grid_steps, grid_span=args.grid_span,
                    refine_steps=args.refine_steps, refine_span=args.refine_span,
                    use_cuda=args.cuda, seed=args.seed,
                    const_name=cname, restricted=args.restricted_phi
                )
                res["int_bits"] = bits
                res["int_angle_mean"] = float("nan")
                res["int_angle_std"] = float("nan")
            else:
                res = run_trials_for_bits_one_const(
                    dim=args.dim, mean=args.mean, std=args.std, trials=args.trials,
                    phi_bits=8, int_bits=int_bits,
                    grid_steps=args.grid_steps, grid_span=args.grid_span,
                    refine_steps=args.refine_steps, refine_span=args.refine_span,
                    use_cuda=args.cuda, seed=args.seed,
                    const_name=cname, restricted=args.restricted_phi
                )
                res["phi_bits"] = bits
                res["phi_angle_mean"] = float("nan")
                res["phi_angle_std"] = float("nan")
                res["rphi_angle_mean"] = float("nan")
                res["rphi_angle_std"] = float("nan")

            res["bits"] = bits
            records.append(res)

    df = pd.DataFrame(records)

    # Plot: for each constant, draw phi mean±std; int is common (same across consts), so plot once
    import matplotlib.pyplot as plt
    fig = plt.figure()

    bits_sorted = sorted(df["bits"].unique())

    # Plot int once (mean across constants, ignoring NaNs)
    df_int = df.groupby("bits", as_index=True)[["int_angle_mean","int_angle_std"]].mean(numeric_only=True)
    plt.errorbar(bits_sorted, df_int["int_angle_mean"].values, yerr=df_int["int_angle_std"].values,
                 fmt='s--', capsize=3, label="int angle (mean ± std)")

    # Plot phi per constant
    for cname in consts:
        d = df[df["const_name"] == cname].set_index("bits").reindex(bits_sorted)
        plt.errorbar(bits_sorted, d["phi_angle_mean"].values, yerr=d["phi_angle_std"].values,
                     fmt='o-', capsize=3, label=f"phi {cname} (mean ± std)")

        if args.restricted_phi:
            plt.errorbar(bits_sorted, d["rphi_angle_mean"].values, yerr=d["rphi_angle_std"].values,
                         fmt='^-', capsize=3, label=f"restricted-phi {cname} (mean ± std)")

    if args.logy:
        plt.yscale("log")

    plt.xlabel("bits")
    plt.ylabel("angle (degrees)")
    title_extra = " (log y)" if args.logy else ""
    plt.title(f"Angle vs Bits: int vs phi (per-constant){title_extra}")
    plt.legend(loc="best")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    import os
    plt.tight_layout()
    out_png = args.out_png
    plt.savefig(out_png, dpi=150)
    try:
        plt.show()
    except Exception:
        pass

    # CSV long-form
    if args.csv:
        df.to_csv(args.csv, index=False)

    print("\nSaved:", out_png)
    if args.csv:
        print("CSV:", args.csv)

    # Print compact summaries
    print("\n=== Summary (bits; mean_deg ± std_deg) ===")
    for cname in consts:
        print(f"\n-- Constant: {cname} --")
        d = df[df["const_name"] == cname].sort_values("bits")
        for _, row in d.iterrows():
            b = int(row["bits"])
            pm, ps = row["phi_angle_mean"], row["phi_angle_std"]
            rm, rs = row.get("rphi_angle_mean", float('nan')), row.get("rphi_angle_std", float('nan'))
            im, istd = row["int_angle_mean"], row["int_angle_std"]
            if args.restricted_phi:
                print(f"bits={b:2d} | phi: {pm:8.4f} ± {ps:7.4f} | rphi: {rm:8.4f} ± {rs:7.4f} | int: {im:8.4f} ± {istd:7.4f}")
            else:
                print(f"bits={b:2d} | phi: {pm:8.4f} ± {ps:7.4f} | int: {im:8.4f} ± {istd:7.4f}")


if __name__ == "__main__":
    main()

