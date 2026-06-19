import argparse
import random
from typing import List, Tuple


def extend_symbols(required: int, symbols: str, avoid: str) -> str:
    """Return ``symbols`` extended to at least ``required`` characters.

    Additional symbols are generated from ascending Unicode code points
    starting at ``!`` while skipping characters present in ``avoid`` or
    already used in ``symbols``. This prevents overlaps with operator
    tokens and enables arbitrarily large groups.
    """
    used = []
    used_set = set()
    for ch in symbols:
        if ch in avoid or ch in used_set:
            continue
        used.append(ch)
        used_set.add(ch)

    code = ord("!")
    while len(used) < required:
        ch = chr(code)
        code += 1
        if ch in avoid or ch in used_set:
            continue
        used.append(ch)
        used_set.add(ch)
    return "".join(used)


def index_to_symbol(idx: int, symbols: str) -> str:
    if idx < len(symbols):
        return symbols[idx]
    return str(idx)


def cyclic_next(current: int, step: int, order: int, wrap: bool) -> int:
    nxt = current + step
    if wrap:
        return nxt % order
    return nxt


def generate_cyclic(order: int, step: int, length: int, wrap: bool,
                     state_symbols: str, op_symbol: str) -> List[str]:
    """Generate a sequence of operations for a cyclic subgroup."""
    elems = [i * step % order for i in range(order // step)]
    state_symbols = extend_symbols(order, state_symbols, op_symbol)
    current = random.choice(elems)
    lines = []
    for _ in range(length):
        nxt = cyclic_next(current, step, order, wrap)
        lhs = index_to_symbol(current, state_symbols)
        rhs = index_to_symbol(step % order, state_symbols)
        res = index_to_symbol(nxt, state_symbols)
        lines.append(f"{lhs}{op_symbol}{rhs}={res}\n")
        current = nxt if wrap else nxt
    return lines


def dihedral_multiply(elem: Tuple[bool, int], op: str, n: int) -> Tuple[bool, int]:
    is_reflect, k = elem
    if op == "r":  # rotation by 1
        return (is_reflect, (k + 1) % n)
    elif op == "s":  # reflection
        if is_reflect:
            return (False, k)
        else:
            return (True, (-k) % n)
    else:
        raise ValueError("invalid operation")


def generate_dihedral(n: int, length: int, state_symbols: str,
                       rot_symbol: str, ref_symbol: str,
                       rotations_only: bool) -> List[str]:
    elems = [(False, i) for i in range(n)]
    if not rotations_only:
        elems += [(True, i) for i in range(n)]
    state_symbols = extend_symbols(len(elems), state_symbols,
                                   rot_symbol + ref_symbol)
    current = random.choice(elems)
    lines = []
    for _ in range(length):
        op = "r" if rotations_only or random.random() < 0.5 else "s"
        nxt = dihedral_multiply(current, op, n)
        lhs = index_to_symbol(elems.index(current), state_symbols)
        rhs = rot_symbol if op == "r" else ref_symbol
        res = index_to_symbol(elems.index(nxt), state_symbols)
        lines.append(f"{lhs}{rhs}={res}\n")
        current = nxt
    return lines


def validate_file(path: str) -> None:
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "=" not in line:
            raise ValueError(f"Invalid line: {line.strip()}")
    print("File looks valid (basic format check).")


def main():
    ap = argparse.ArgumentParser(description="Group theory dataset generator")
    ap.add_argument("--group", choices=["cyclic", "dihedral"], default="cyclic")
    ap.add_argument("--order", type=int, default=3,
                    help="group order or polygon sides")
    ap.add_argument("--length", type=int, default=100,
                    help="number of operations to generate")
    ap.add_argument("--step", type=int, default=1,
                    help="generator step for cyclic groups")
    ap.add_argument("--no-closure", action="store_true",
                    help="disable wrap around for cyclic groups")
    ap.add_argument("--state-symbols", type=str,
                    default="abcdefghijklmnopqrstuvwxyz",
                    help="symbols for group elements")
    ap.add_argument("--operator-symbol", type=str, default="+",
                    help="symbol for the cyclic operator")
    ap.add_argument("--rot-symbol", type=str, default="*",
                    help="symbol for dihedral rotation")
    ap.add_argument("--ref-symbol", type=str, default="/",
                    help="symbol for dihedral reflection")
    ap.add_argument("--rotations-only", action="store_true",
                    help="use only rotational subgroup in dihedral groups")
    ap.add_argument("--output", type=str, default="dataset.txt")
    ap.add_argument("--validate", type=str,
                    help="validate an existing dataset file")
    args = ap.parse_args()

    if args.validate:
        validate_file(args.validate)
        return

    random.seed(0)
    if args.group == "cyclic":
        lines = generate_cyclic(args.order, args.step, args.length,
                                not args.no_closure, args.state_symbols,
                                args.operator_symbol)
    else:
        lines = generate_dihedral(args.order, args.length, args.state_symbols,
                                  args.rot_symbol, args.ref_symbol,
                                  args.rotations_only)
    with open(args.output, "w") as f:
        f.writelines(lines)
    print(f"wrote {len(lines)} lines to {args.output}")


if __name__ == "__main__":
    main()
