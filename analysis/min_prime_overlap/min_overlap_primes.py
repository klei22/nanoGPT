
#!/usr/bin/env python3
"""
Find n distinct prime "cycle lengths" whose first overlap (LCM) is
no sooner than threshold t, while minimizing the sum of the primes.

For distinct primes p1<...<pn:
    LCM(p1,...,pn) = p1*p2*...*pn

We search for:
    minimize sum(pi)
    subject to product(pi) >= t   (or > t if --strict)

Notes:
- Duplicating a prime never helps LCM, so we assume distinct primes.
- Exact solution via branch-and-bound.
- Uses deterministic Miller-Rabin for 64-bit integers (and a reasonable fallback for bigger).
"""

from __future__ import annotations

import argparse
import math
from functools import lru_cache
from typing import List, Tuple


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

    # Deterministic Miller-Rabin bases for unsigned 64-bit integers:
    # See commonly used set: [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
    if n < 2**64:
        bases = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
    else:
        # Fallback: not guaranteed deterministic for arbitrarily huge n,
        # but works well in practice for typical thresholds.
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


def min_sum_primes_overlap(n: int, t: int, strict: bool = False) -> Tuple[List[int], int, int]:
    """
    Return (best_primes, best_sum, overlap_lcm).

    If strict=False, require overlap >= t.
    If strict=True, require overlap > t.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if t < 1:
        raise ValueError("t must be >= 1")

    target = t + 1 if strict else t

    # If the n smallest primes already meet the threshold, they are automatically optimal
    smallest = list_smallest_primes_ge(2, n)
    prod_cap = 1
    for p in smallest:
        prod_cap *= p
        if prod_cap >= target:
            prod_cap = target
            break
    if prod_cap >= target:
        return list(smallest), sum(smallest), math.prod(smallest)

    # Initial feasible upper bound:
    # take n consecutive primes starting at the first prime >= ceil(target^(1/n))
    r = int_nth_root_ceil(target, n)
    p0 = next_prime_ge(r)
    cand = [p0]
    while len(cand) < n:
        cand.append(next_prime_ge(cand[-1] + 1))

    best = {"sum": sum(cand), "set": cand[:]}

    def dfs(min_start: int, chosen: List[int], current_sum: int, current_prod_cap: int) -> None:
        k = len(chosen)
        remaining = n - k

        # Safe lower bound on achievable sum from this node:
        lb_sum = current_sum + sum_smallest_primes_ge(min_start, remaining)
        if lb_sum >= best["sum"]:
            return

        # If we already meet the product threshold, the best completion is to add the smallest remaining primes.
        if current_prod_cap >= target:
            completion = list_smallest_primes_ge(min_start, remaining)
            total_sum = current_sum + sum(completion)
            if total_sum < best["sum"]:
                best["sum"] = total_sum
                best["set"] = chosen + list(completion)
            return

        # Iterate candidate next primes in increasing order
        p = next_prime_ge(min_start)
        while True:
            # Monotone "can we still beat best?" bound:
            # If we pick p now, the smallest possible completion uses the next (remaining-1) primes after p.
            min_comp_sum = current_sum + p + sum_smallest_primes_ge(p + 1, remaining - 1)
            if min_comp_sum >= best["sum"]:
                break  # larger p will only worsen this bound

            new_sum = current_sum + p
            new_prod_cap = current_prod_cap * p
            if new_prod_cap >= target:
                new_prod_cap = target

            if remaining == 1:
                # Leaf
                if new_prod_cap >= target and new_sum < best["sum"]:
                    best["sum"] = new_sum
                    best["set"] = chosen + [p]
            else:
                m = remaining - 1
                required = 1 if new_prod_cap >= target else ceil_div(target, new_prod_cap)

                # Optional (safe) feasibility pruning:
                # With sum budget S = best_sum - new_sum for the remaining m primes, the maximum
                # product (relaxing to positive real numbers) is at most (S/m)^m.
                # If even that can't reach 'required', this branch can't succeed while beating best.
                budget = best["sum"] - new_sum
                feasible = True
                if required > 1:
                    if budget <= 0:
                        feasible = False
                    else:
                        # Compare logs to avoid overflow:
                        # log((budget/m)^m) = m*(log(budget) - log(m))
                        log_max = m * (math.log(budget) - math.log(m))
                        feasible = log_max + 1e-12 >= math.log(required)

                if feasible:
                    dfs(p + 1, chosen + [p], new_sum, new_prod_cap)

            p = next_prime_ge(p + 1)

    dfs(2, [], 0, 1)

    best_set = best["set"]
    return best_set, best["sum"], math.prod(best_set)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Find n distinct primes with minimum sum such that their overlap (LCM) is no sooner than t."
    )
    ap.add_argument("-n", "--count", type=int, required=True, help="Number of primes (n).")
    ap.add_argument("-t", "--threshold", type=int, required=True, help="Overlap threshold (t).")
    ap.add_argument("--strict", action="store_true", help="Require overlap > t (instead of >= t).")

    args = ap.parse_args()

    primes, s, overlap = min_sum_primes_overlap(args.count, args.threshold, strict=args.strict)
    cmp = ">" if args.strict else ">="

    print(f"n={args.count}, threshold t={args.threshold} (require overlap {cmp} t)")
    print(f"best primes: {primes}")
    print(f"sum: {s}")
    print(f"overlap (LCM): {overlap}")


if __name__ == "__main__":
    main()

