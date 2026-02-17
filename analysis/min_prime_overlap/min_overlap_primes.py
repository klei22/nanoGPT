#!/usr/bin/env python3
"""
Min-sum primes whose first "overlap" (LCM) is no sooner than threshold t.

Interpretation (matches the 3 and 5 -> 15 example):
- Two periods "overlap again" at their LCM.
- For distinct primes p1 < ... < pk, LCM(p1,...,pk) = p1*p2*...*pk.

This script supports:
1) Exact count mode: choose exactly n primes, minimize sum, subject to overlap >= t (or > t with --strict).
2) Up-to mode: choose at most N primes (k <= N), minimize sum, subject to overlap >= t (or > t).

Plus optional plotting with matplotlib:
- Sum vs count (exact-k and best-<=k)
- Overlap vs count (log scale)
- Solution composition (bar chart) and cumulative overlap growth
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Small primes used for quick divisibility checks and as fallback MR bases
_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_probable_prime(n: int) -> bool:
    """Fast primality test: trial division by small primes + Miller-Rabin."""
    if n < 2:
        return False

    for p in _SMALL_PRIMES:
        if n % p == 0:
            return n == p

    # Write n-1 = d * 2^s with d odd
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    # Deterministic Miller-Rabin bases for unsigned 64-bit integers
    # Commonly used proven set for n < 2^64:
    # [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
    if n < 2**64:
        bases = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
    else:
        # Fallback: not guaranteed deterministic for arbitrarily huge integers,
        # but fine for most practical thresholds.
        bases = _SMALL_PRIMES

    for a in bases:
        a %= n
        if a == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False

    return True


@lru_cache(maxsize=None)
def next_prime_ge(n: int) -> int:
    """Return the smallest prime >= n."""
    if n <= 2:
        return 2
    if n % 2 == 0:
        n += 1
    while True:
        if is_probable_prime(n):
            return n
        n += 2


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def int_nth_root_ceil(a: int, n: int) -> int:
    """Smallest integer x such that x**n >= a (a>=0, n>=1)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if a <= 1:
        return 1

    lo, hi = 1, 2
    while pow(hi, n) < a:
        hi *= 2

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if pow(mid, n) >= a:
            hi = mid
        else:
            lo = mid
    return hi


@lru_cache(maxsize=None)
def sum_smallest_primes_ge(start: int, count: int) -> int:
    """Sum of the 'count' smallest distinct primes >= start."""
    s = 0
    x = start
    for _ in range(count):
        p = next_prime_ge(x)
        s += p
        x = p + 1
    return s


@lru_cache(maxsize=None)
def list_smallest_primes_ge(start: int, count: int) -> Tuple[int, ...]:
    """Tuple of the 'count' smallest distinct primes >= start."""
    res: List[int] = []
    x = start
    for _ in range(count):
        p = next_prime_ge(x)
        res.append(p)
        x = p + 1
    return tuple(res)


@dataclass(frozen=True)
class Solution:
    count: int
    primes: Tuple[int, ...]
    sum: int
    overlap: int  # LCM; for distinct primes it's the product


def min_sum_primes_overlap_exact(n: int, t: int, strict: bool = False) -> Solution:
    """
    Exact-n mode:
      minimize sum(pi) subject to product(pi) >= t (or >t if strict)
      with distinct primes p1<...<pn
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if t < 1:
        raise ValueError("t must be >= 1")

    target = t + 1 if strict else t

    # n=1 is trivial: smallest prime >= target
    if n == 1:
        p = next_prime_ge(target)
        return Solution(count=1, primes=(p,), sum=p, overlap=p)

    # If the n smallest primes already meet target, they're optimal
    smallest = list_smallest_primes_ge(2, n)
    prod_cap = 1
    for p in smallest:
        prod_cap *= p
        if prod_cap >= target:
            prod_cap = target
            break
    if prod_cap >= target:
        overlap = math.prod(smallest)
        return Solution(count=n, primes=tuple(smallest), sum=sum(smallest), overlap=overlap)

    # Initial feasible upper bound:
    # n consecutive primes starting at prime >= ceil(target^(1/n))
    r = int_nth_root_ceil(target, n)
    p0 = next_prime_ge(r)
    cand = [p0]
    while len(cand) < n:
        cand.append(next_prime_ge(cand[-1] + 1))
    best_sum = sum(cand)
    best_set = tuple(cand)

    def dfs(min_start: int, chosen: List[int], current_sum: int, current_prod_cap: int) -> None:
        nonlocal best_sum, best_set

        k = len(chosen)
        remaining = n - k

        # Lower bound on sum using smallest remaining primes
        lb_sum = current_sum + sum_smallest_primes_ge(min_start, remaining)
        if lb_sum >= best_sum:
            return

        # If already hit target, fill rest minimally
        if current_prod_cap >= target:
            completion = list_smallest_primes_ge(min_start, remaining)
            total_sum = current_sum + sum(completion)
            if total_sum < best_sum:
                best_sum = total_sum
                best_set = tuple(chosen + list(completion))
            return

        p = next_prime_ge(min_start)
        while True:
            # If we pick p now, minimal completion after p
            min_comp_sum = current_sum + p + sum_smallest_primes_ge(p + 1, remaining - 1)
            if min_comp_sum >= best_sum:
                break  # larger p only increases this

            new_sum = current_sum + p
            new_prod_cap = current_prod_cap * p
            if new_prod_cap >= target:
                new_prod_cap = target

            if remaining == 1:
                if new_prod_cap >= target and new_sum < best_sum:
                    best_sum = new_sum
                    best_set = tuple(chosen + [p])
            else:
                m = remaining - 1
                required = 1 if new_prod_cap >= target else ceil_div(target, new_prod_cap)

                # Feasibility pruning (AM-GM upper bound under sum budget)
                budget = best_sum - new_sum
                feasible = True
                if required > 1:
                    if budget <= 0:
                        feasible = False
                    else:
                        # max product of m positive reals with sum=budget is (budget/m)^m
                        log_max = m * (math.log(budget) - math.log(m))
                        feasible = log_max + 1e-12 >= math.log(required)

                if feasible:
                    dfs(p + 1, chosen + [p], new_sum, new_prod_cap)

            p = next_prime_ge(p + 1)

    dfs(2, [], 0, 1)

    overlap = math.prod(best_set)
    return Solution(count=n, primes=best_set, sum=best_sum, overlap=overlap)


def min_sum_primes_overlap_upto(
    max_n: int,
    t: int,
    strict: bool = False,
    min_n: int = 1,
) -> Tuple[Solution, List[Solution]]:
    """
    Up-to mode:
      choose any k in [min_n, max_n] that minimizes sum subject to overlap constraint.

    Returns:
      (best_solution_over_all_k, list_of_exact_k_solutions)
    """
    if max_n < 1:
        raise ValueError("max_n must be >= 1")
    if not (1 <= min_n <= max_n):
        raise ValueError("min_n must satisfy 1 <= min_n <= max_n")

    all_solutions: List[Solution] = []
    best: Optional[Solution] = None

    for k in range(min_n, max_n + 1):
        sol = min_sum_primes_overlap_exact(k, t, strict=strict)
        all_solutions.append(sol)

        if best is None:
            best = sol
        else:
            # Primary: smaller sum
            # Tie-breaker: fewer primes (simpler)
            if (sol.sum < best.sum) or (sol.sum == best.sum and sol.count < best.count):
                best = sol

    assert best is not None
    return best, all_solutions


# -------------------------
# Plotting
# -------------------------

def _ensure_dir(d: Optional[str]) -> Optional[Path]:
    if not d:
        return None
    p = Path(d).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _maybe_savefig(fig, save_dir: Optional[Path], filename: str) -> None:
    if save_dir is None:
        return
    fig.savefig(save_dir / filename, dpi=200, bbox_inches="tight")


def plot_counts_curves(
    per_k: List[Solution],
    t: int,
    strict: bool,
    save_dir: Optional[str] = None,
    prefix: str = "overlap",
    show: bool = True,
) -> None:
    """
    Two plots:
      1) Sum vs k: exact-k and best-<=k
      2) Overlap vs k (log y)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Plotting requires matplotlib. Install with: pip install matplotlib")
        return

    save_path = _ensure_dir(save_dir)
    cmp = ">" if strict else ">="
    target = t + 1 if strict else t

    ks = [s.count for s in per_k]
    sums = [s.sum for s in per_k]
    overlaps = [s.overlap for s in per_k]

    # best up-to-k curve
    best_upto: List[int] = []
    cur = float("inf")
    for s in sums:
        cur = min(cur, s)
        best_upto.append(int(cur))

    # Plot 1: Sum vs k
    fig1 = plt.figure()
    plt.plot(ks, sums, marker="o", label="Best sum (exact k)")
    plt.plot(ks, best_upto, marker="o", linestyle="--", label="Best sum (≤ k)")
    plt.xlabel("Number of primes (k)")
    plt.ylabel("Sum of primes")
    plt.title(f"Minimum sum vs k (overlap {cmp} {t:,})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _maybe_savefig(fig1, save_path, f"{prefix}_sum_vs_k.png")
    if show:
        plt.show()
    plt.close(fig1)

    # Plot 2: Overlap vs k (log scale)
    fig2 = plt.figure()
    plt.plot(ks, overlaps, marker="o")
    plt.yscale("log")
    plt.axhline(target, linestyle="--")
    plt.xlabel("Number of primes (k)")
    plt.ylabel("Overlap / LCM (log scale)")
    plt.title(f"Overlap vs k (threshold line at {target:,})")
    plt.grid(True, which="both", alpha=0.3)
    _maybe_savefig(fig2, save_path, f"{prefix}_overlap_vs_k.png")
    if show:
        plt.show()
    plt.close(fig2)


def plot_solution_details(
    sol: Solution,
    t: int,
    strict: bool,
    save_dir: Optional[str] = None,
    prefix: str = "overlap",
    show: bool = True,
) -> None:
    """
    Two plots about the chosen solution:
      1) Bar chart of primes
      2) Cumulative log10(overlap) as you add primes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Plotting requires matplotlib. Install with: pip install matplotlib")
        return

    save_path = _ensure_dir(save_dir)
    target = t + 1 if strict else t

    primes = list(sol.primes)
    idx = list(range(1, len(primes) + 1))

    # Plot 1: bar chart of primes
    fig1 = plt.figure()
    plt.bar(idx, primes)
    plt.xlabel("Prime index in solution")
    plt.ylabel("Prime value")
    plt.title(f"Chosen primes (k={sol.count}, sum={sol.sum:,}, overlap={sol.overlap:,})")
    plt.grid(True, axis="y", alpha=0.3)
    _maybe_savefig(fig1, save_path, f"{prefix}_solution_primes.png")
    if show:
        plt.show()
    plt.close(fig1)

    # Plot 2: cumulative log10 overlap growth
    cum_log10: List[float] = []
    prod = 1
    for p in primes:
        prod *= p
        cum_log10.append(math.log10(prod))

    fig2 = plt.figure()
    plt.plot(idx, cum_log10, marker="o")
    plt.axhline(math.log10(target), linestyle="--")
    plt.xlabel("Number of primes included")
    plt.ylabel("log10(cumulative overlap)")
    plt.title("How overlap grows as primes are added")
    plt.grid(True, alpha=0.3)
    _maybe_savefig(fig2, save_path, f"{prefix}_solution_cum_overlap.png")
    if show:
        plt.show()
    plt.close(fig2)


def write_counts_csv(per_k: List[Solution], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k", "sum", "overlap", "primes"])
        for s in per_k:
            w.writerow([s.count, s.sum, s.overlap, " ".join(map(str, s.primes))])


# -------------------------
# CLI
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Find primes minimizing sum s.t. overlap (LCM) is no sooner than threshold t."
    )

    ap.add_argument("-t", "--threshold", type=int, required=True, help="Threshold (t).")
    ap.add_argument("--strict", action="store_true", help="Require overlap > t (instead of >= t).")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("-n", "--count", type=int, help="Exact number of primes (n).")
    mode.add_argument("--max-count", type=int, help="Maximum number of primes allowed (k ≤ max-count).")

    ap.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum primes to allow in up-to mode (default 1).",
    )

    ap.add_argument(
        "--table",
        action="store_true",
        help="Print the per-k solutions table (useful with --max-count or plots).",
    )

    ap.add_argument(
        "--plot",
        choices=["none", "counts", "solution", "both"],
        default="none",
        help="Create plots (requires matplotlib).",
    )
    ap.add_argument(
        "--plot-max-k",
        type=int,
        default=None,
        help="When plotting counts curves, compute exact-k solutions for k=1..plot-max-k (defaults to count/max-count).",
    )
    ap.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save PNG plots (optional). If omitted, plots are only shown.",
    )
    ap.add_argument(
        "--prefix",
        type=str,
        default="overlap",
        help="Filename prefix for saved plots.",
    )
    ap.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show() (useful on servers/headless).",
    )

    ap.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Write per-k results to CSV at this path (only if per-k results are computed).",
    )

    args = ap.parse_args()

    t = args.threshold
    strict = args.strict
    cmp = ">" if strict else ">="
    target = t + 1 if strict else t

    # Solve main request
    if args.count is not None:
        sol = min_sum_primes_overlap_exact(args.count, t, strict=strict)
        print(f"Mode: exact count n={args.count}")
        print(f"Constraint: overlap {cmp} t (i.e. overlap >= {target:,})")
        print(f"Best primes: {list(sol.primes)}")
        print(f"Sum: {sol.sum:,}")
        print(f"Overlap (LCM): {sol.overlap:,}")

        best = sol
        per_k: Optional[List[Solution]] = None

    else:
        if args.max_count < 1:
            raise SystemExit("max-count must be >= 1")
        if not (1 <= args.min_count <= args.max_count):
            raise SystemExit("min-count must satisfy 1 <= min-count <= max-count")

        best, per_list = min_sum_primes_overlap_upto(
            args.max_count, t, strict=strict, min_n=args.min_count
        )

        print(f"Mode: up-to (min-count={args.min_count}, max-count={args.max_count})")
        print(f"Constraint: overlap {cmp} t (i.e. overlap >= {target:,})")
        print()
        print(f"Best k: {best.count}")
        print(f"Best primes: {list(best.primes)}")
        print(f"Sum: {best.sum:,}")
        print(f"Overlap (LCM): {best.overlap:,}")

        per_k = per_list

    # Optionally compute per-k for plots/table even in exact-n mode
    if args.plot in ("counts", "both") or args.table or args.csv:
        if args.plot_max_k is not None:
            max_k = args.plot_max_k
        else:
            max_k = args.count if args.count is not None else args.max_count

        # In exact-n mode, we default to computing k=1..n for visualization
        # In up-to mode, also respect --min-count.
        min_k = 2
        if args.max_count is not None:
            min_k = max(min_k, args.min_count)

        if min_k > max_k:
            raise SystemExit(f"plot min_k ({min_k}) is > max_k ({max_k}); nothing to plot.")

        per_k = []
        for k in range(min_k, max_k + 1):
            per_k.append(min_sum_primes_overlap_exact(k, t, strict=strict))


        if args.table:
            print("\nPer-k best (exact k) results:")
            for s in per_k:
                print(f"  k={s.count:>2}  sum={s.sum:>8,}  overlap={s.overlap:>12,}  primes={list(s.primes)}")

        if args.csv:
            write_counts_csv(per_k, args.csv)
            print(f"\nWrote CSV: {args.csv}")

        # Plotting
        show = not args.no_show
        if args.plot in ("counts", "both"):
            plot_counts_curves(
                per_k=per_k,
                t=t,
                strict=strict,
                save_dir=args.save_dir,
                prefix=args.prefix,
                show=show,
            )

    if args.plot in ("solution", "both"):
        show = not args.no_show
        plot_solution_details(
            sol=best,
            t=t,
            strict=strict,
            save_dir=args.save_dir,
            prefix=args.prefix,
            show=show,
        )


if __name__ == "__main__":
    main()

