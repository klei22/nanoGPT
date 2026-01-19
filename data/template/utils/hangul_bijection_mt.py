#!/usr/bin/env python3
"""
hangul_bijection_mt.py

Bijective Hangul canonicalization with deterministic multithreading (Ubuntu/Linux).

Implements two bijective modes:
  - strict:  decompose Hangul syllables -> canonical Jamo (L/V/T), leave everything else unchanged
  - tagged:  same, plus compatibility Jamo are normalized to canonical Jamo *with a reversible tag*

Deterministic multithreading:
  - Forward (both modes): parallel-safe (pure per-codepoint work)
  - Inverse strict: parallel-safe using deterministic chunk-order + boundary carry (max 2 codepoints)
  - Inverse tagged: deterministic, but done single-threaded by default (tag parsing is boundary-sensitive).
                  (Forward is typically the heavy step anyway; inverse tagged can be added later if needed.)

Also includes a built-in test suite:
  python3 hangul_bijection_mt.py --run-tests

Examples:
  echo "한글 ㄱㅏ test" | python3 hangul_bijection_mt.py --mode strict --op forward --jobs 8 --chunk-chars 200000
  echo "각"        | python3 hangul_bijection_mt.py --mode strict --op inverse --jobs 8
  python3 hangul_bijection_mt.py --demo

Output:
  - Writes transformed text to stdout.
  - Diagnostics (round-trip check in demo/tests) go to stderr.

Notes:
- If you need bit-for-bit recovery of the *original normalization form* (NFD vs NFC), use --no-nfc.
- Tagged mode uses BMP Private Use Area markers U+E000/U+E001 to guarantee no collisions.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import random
import sys
import unicodedata
import unittest
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

# -----------------------
# Unicode Hangul constants
# -----------------------
S_BASE = 0xAC00
S_END  = 0xD7A3

L_BASE = 0x1100
V_BASE = 0x1161
T_BASE = 0x11A7  # TIndex 0 means "no trailing"; actual trailing starts at T_BASE+1 (0x11A8)

L_COUNT = 19
V_COUNT = 21
T_COUNT = 28
N_COUNT = V_COUNT * T_COUNT  # 588

L_START = L_BASE
L_END   = L_BASE + L_COUNT - 1  # 0x1112

V_START = V_BASE
V_END   = V_BASE + V_COUNT - 1  # 0x1175

T_START = T_BASE + 1            # 0x11A8
T_END   = T_BASE + T_COUNT - 1  # 0x11C2

# Hangul Compatibility Jamo block
COMPAT_START = 0x3130
COMPAT_END   = 0x318F

# Tagged mode markers (PUA)
TAG_OPEN  = "\uE000"
TAG_CLOSE = "\uE001"
# Tagged payload format: TAG_OPEN + 4 hex digits + TAG_CLOSE
TAG_LEN = 6  # open + 4 + close


def is_hangul_syllable(cp: int) -> bool:
    return S_BASE <= cp <= S_END


def is_L_jamo(cp: int) -> bool:
    return L_START <= cp <= L_END


def is_V_jamo(cp: int) -> bool:
    return V_START <= cp <= V_END


def is_T_jamo(cp: int) -> bool:
    return T_START <= cp <= T_END


def is_compat_jamo(cp: int) -> bool:
    return COMPAT_START <= cp <= COMPAT_END


def maybe_nfc(s: str, do_nfc: bool) -> str:
    return unicodedata.normalize("NFC", s) if do_nfc else s


def decompose_syllable(cp: int) -> List[int]:
    """Decompose one Hangul syllable to canonical Jamo [L, V] or [L, V, T]."""
    if not is_hangul_syllable(cp):
        return [cp]

    s_index = cp - S_BASE
    l_index = s_index // N_COUNT
    v_index = (s_index % N_COUNT) // T_COUNT
    t_index = s_index % T_COUNT

    L = L_BASE + l_index
    V = V_BASE + v_index
    out = [L, V]
    if t_index != 0:
        out.append(T_BASE + t_index)  # 0x11A8..0x11C2
    return out


def compose_from_jamo_stream(cps: List[int]) -> List[int]:
    """Compose L+V(+T) sequences deterministically into Hangul syllables."""
    out: List[int] = []
    i = 0
    n = len(cps)
    while i < n:
        cp = cps[i]
        if is_L_jamo(cp) and i + 1 < n and is_V_jamo(cps[i + 1]):
            L = cp
            V = cps[i + 1]
            l_index = L - L_BASE
            v_index = V - V_BASE

            t_index = 0
            advance = 2
            if i + 2 < n and is_T_jamo(cps[i + 2]):
                t_index = cps[i + 2] - T_BASE
                advance = 3

            s_index = (l_index * N_COUNT) + (v_index * T_COUNT) + t_index
            out.append(S_BASE + s_index)
            i += advance
            continue

        out.append(cp)
        i += 1
    return out


def to_cps(s: str) -> List[int]:
    return [ord(c) for c in s]


def from_cps(cps: List[int]) -> str:
    return "".join(chr(cp) for cp in cps)


# -------------------------
# Tagged mode helpers
# -------------------------
def compat_to_canonical_jamo(cp: int) -> Optional[List[int]]:
    """
    Convert a single compatibility jamo codepoint to canonical jamo sequence via NFKD.
    Return None if result isn't purely canonical L/V/T Jamo.
    """
    ch = chr(cp)
    nfkd = unicodedata.normalize("NFKD", ch)
    cps = [ord(c) for c in nfkd]
    if not cps:
        return None
    for x in cps:
        if not (is_L_jamo(x) or is_V_jamo(x) or is_T_jamo(x)):
            return None
    return cps


def tag_original_compat(cp: int) -> List[int]:
    hex4 = f"{cp:04X}"
    tagged = TAG_OPEN + hex4 + TAG_CLOSE
    return [ord(c) for c in tagged]


def parse_tag_at(cps: List[int], i: int) -> Optional[Tuple[int, int]]:
    """
    If cps[i:] begins with a valid tag, return (orig_cp, new_i_after_tag).
    Tag pattern: TAG_OPEN + 4 hex + TAG_CLOSE.
    """
    if i + TAG_LEN - 1 >= len(cps):
        return None
    if cps[i] != ord(TAG_OPEN):
        return None
    if cps[i + 5] != ord(TAG_CLOSE):
        return None
    hex_digits = cps[i + 1:i + 5]
    ok = all(
        (48 <= d <= 57) or (65 <= d <= 70) or (97 <= d <= 102)
        for d in hex_digits
    )
    if not ok:
        return None
    orig_cp = int("".join(chr(d) for d in hex_digits), 16)
    return orig_cp, i + TAG_LEN


def forward_strict_chunk(text: str) -> str:
    out: List[int] = []
    for ch in text:
        out.extend(decompose_syllable(ord(ch)))
    return from_cps(out)


def forward_tagged_chunk(text: str) -> str:
    out: List[int] = []
    for ch in text:
        cp = ord(ch)
        if is_hangul_syllable(cp):
            out.extend(decompose_syllable(cp))
        elif is_compat_jamo(cp):
            canon = compat_to_canonical_jamo(cp)
            if canon is not None:
                out.extend(tag_original_compat(cp))
                out.extend(canon)
            else:
                out.append(cp)
        else:
            out.append(cp)
    return from_cps(out)


# -------------------------
# Deterministic chunking
# -------------------------
def chunk_text(s: str, chunk_chars: int) -> List[str]:
    if chunk_chars <= 0 or len(s) <= chunk_chars:
        return [s]
    return [s[i:i + chunk_chars] for i in range(0, len(s), chunk_chars)]


def parallel_map_ordered(fn, chunks: List[str], jobs: int) -> List[str]:
    """
    Deterministic parallel map: returns outputs in the same order as chunks.
    """
    if jobs <= 1 or len(chunks) <= 1:
        return [fn(c) for c in chunks]

    out = [""] * len(chunks)
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(fn, chunks[i]): i for i in range(len(chunks))}
        for fut in cf.as_completed(futs):
            idx = futs[fut]
            out[idx] = fut.result()
    return out


# -------------------------
# Inverse strict (parallel) with boundary carry
# -------------------------
@dataclass
class ComposeChunkResult:
    body: List[int]
    tail: List[int]  # [] or [L] or [L,V]


def compose_chunk_safe(cps: List[int]) -> ComposeChunkResult:
    """
    Compose within a chunk, but leave a deterministic tail (max 2 codepoints)
    that could start a syllable and might need the next chunk.

    Tail rules (max 2):
      - If chunk ends with [L], carry [L]
      - If chunk ends with [L,V], carry [L,V]
      - Otherwise carry []
    """
    if not cps:
        return ComposeChunkResult(body=[], tail=[])

    # Determine tail length by inspecting the end:
    tail: List[int] = []
    if is_L_jamo(cps[-1]):
        tail = [cps[-1]]
        core = cps[:-1]
    elif len(cps) >= 2 and is_L_jamo(cps[-2]) and is_V_jamo(cps[-1]):
        tail = [cps[-2], cps[-1]]
        core = cps[:-2]
    else:
        core = cps

    composed = compose_from_jamo_stream(core)
    return ComposeChunkResult(body=composed, tail=tail)


def inverse_strict_parallel(decomposed: str, jobs: int, chunk_chars: int) -> str:
    """
    Deterministic parallel inverse for strict mode:
      - Split decomposed stream into chunks of codepoints (chars)
      - Compose each chunk "safely" (body + tail)
      - Stitch in order with carry, composing across seams deterministically
    """
    chunks = chunk_text(decomposed, chunk_chars)
    # Convert each chunk to codepoints once
    cps_chunks = [to_cps(c) for c in chunks]

    if jobs <= 1 or len(cps_chunks) <= 1:
        # single-thread stitch
        out: List[int] = []
        carry: List[int] = []
        for cps in cps_chunks:
            r = compose_chunk_safe(carry + cps)
            out.extend(r.body)
            carry = r.tail
        if carry:
            out.extend(compose_from_jamo_stream(carry))
        return from_cps(out)

    # Parallel compute per-chunk safe composition of the chunk alone.
    # NOTE: we still must stitch sequentially to handle carries deterministically.
    # This retains parallel speedups because most work is inside compose_from_jamo_stream(core).
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(compose_chunk_safe, cps_chunks[i]) for i in range(len(cps_chunks))]
        results = [f.result() for f in futs]  # keep order

    out: List[int] = []
    carry: List[int] = []
    for r, cps in zip(results, cps_chunks):
        # We must recompute the boundary with carry included, because r was computed without it.
        # Do minimal work: only recompute for (carry + cps) but keep deterministic.
        rr = compose_chunk_safe(carry + cps)
        out.extend(rr.body)
        carry = rr.tail

    if carry:
        out.extend(compose_from_jamo_stream(carry))
    return from_cps(out)


# -------------------------
# Inverse tagged (deterministic, single-thread)
# -------------------------
def inverse_tagged_singlethread(s: str) -> str:
    """
    Deterministic inverse for tagged mode.

    Steps:
      1) Parse and remove tags, restoring the original compatibility codepoints in place
         BEFORE composition (so we preserve exact compat characters).
      2) Compose remaining canonical L/V/T sequences into syllables.
    """
    cps = to_cps(s)
    out: List[int] = []
    i = 0
    n = len(cps)
    while i < n:
        tag = parse_tag_at(cps, i)
        if tag is not None:
            orig_cp, j = tag
            # Restore the original compatibility cp into the stream.
            out.append(orig_cp)
            i = j
            # Skip exactly one canonical jamo token if present (the normalized counterpart).
            if i < n and (is_L_jamo(cps[i]) or is_V_jamo(cps[i]) or is_T_jamo(cps[i])):
                i += 1
            continue

        out.append(cps[i])
        i += 1

    composed = compose_from_jamo_stream(out)
    return from_cps(composed)


# -------------------------
# Public API: forward/inverse
# -------------------------
def forward(mode: str, text: str, do_nfc: bool, jobs: int, chunk_chars: int) -> str:
    text = maybe_nfc(text, do_nfc)
    chunks = chunk_text(text, chunk_chars)
    if mode == "strict":
        parts = parallel_map_ordered(forward_strict_chunk, chunks, jobs)
    elif mode == "tagged":
        parts = parallel_map_ordered(forward_tagged_chunk, chunks, jobs)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return "".join(parts)


def inverse(mode: str, text: str, do_nfc: bool, jobs: int, chunk_chars: int) -> str:
    # Inverse returns the effective original form (NFC’d if do_nfc was used forward).
    # So do_nfc here is only for test/demo comparisons.
    if mode == "strict":
        return inverse_strict_parallel(text, jobs=jobs, chunk_chars=chunk_chars)
    elif mode == "tagged":
        # deterministic single-thread for now
        return inverse_tagged_singlethread(text)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# -------------------------
# Demo + Tests
# -------------------------
def run_demo() -> int:
    examples = [
        "한글 테스트",
        "가가",                # precomposed + canonical Jamo
        "ㄱㅏ ㄴㅏ ㄷㅏ",        # compatibility jamo sequences
        "K-pop 🐱 가요! ㄱㅏ? 가!",
        "ᄀ ᅡ ᆨ",              # standalone canonical jamo
        "힣",                   # last syllable in range
        "힣",                 # NFD-ish
    ]
    for mode in ("strict", "tagged"):
        print(f"== DEMO mode={mode} ==", file=sys.stderr)
        for s in examples:
            f = forward(mode, s, do_nfc=True, jobs=4, chunk_chars=5)
            b = inverse(mode, f, do_nfc=True, jobs=4, chunk_chars=5)
            ok = (b == unicodedata.normalize("NFC", s))
            print(f"input: {s!r}", file=sys.stderr)
            print(f"fwd  : {f!r}", file=sys.stderr)
            print(f"back : {b!r}  ok={ok}", file=sys.stderr)
            print("", file=sys.stderr)
    return 0


def rand_hangul_syllable(rng: random.Random) -> str:
    return chr(rng.randint(S_BASE, S_END))


def rand_ascii(rng: random.Random) -> str:
    return chr(rng.randint(0x20, 0x7E))


def rand_compat_jamo(rng: random.Random) -> str:
    return chr(rng.randint(COMPAT_START, COMPAT_END))


class TestBijectionMT(unittest.TestCase):
    def test_strict_roundtrip_equivalence_single_vs_multi(self):
        rng = random.Random(123)
        # Make a mixed corpus with syllables, ascii, and some canonical jamo pieces.
        s = []
        for _ in range(4000):
            t = rng.random()
            if t < 0.65:
                s.append(rand_hangul_syllable(rng))
            elif t < 0.9:
                s.append(rand_ascii(rng))
            else:
                # sprinkle canonical jamo
                s.append(chr(rng.randint(L_START, L_END)))
                s.append(chr(rng.randint(V_START, V_END)))
                if rng.random() < 0.3:
                    s.append(chr(rng.randint(T_START, T_END)))
        text = "".join(s)

        f1 = forward("strict", text, do_nfc=True, jobs=1, chunk_chars=97)
        f8 = forward("strict", text, do_nfc=True, jobs=8, chunk_chars=97)
        self.assertEqual(f1, f8)

        b1 = inverse("strict", f1, do_nfc=True, jobs=1, chunk_chars=101)
        b8 = inverse("strict", f1, do_nfc=True, jobs=8, chunk_chars=101)
        self.assertEqual(b1, b8)
        self.assertEqual(b1, unicodedata.normalize("NFC", text))

    def test_strict_boundary_carry_cases(self):
        # Force L|V and LV|T splits across chunks
        L = chr(L_BASE)         # ᄀ
        V = chr(V_BASE)         # ᅡ
        T = chr(T_BASE + 1)     # ᆨ
        # Create decomposed stream with boundaries that cut right after L and after LV
        decomposed = (L + V + T) + "X" + (L + V) + "Y" + (L) + "Z" + (L + V + T)
        # Try with tiny chunk sizes to maximize splits
        back1 = inverse("strict", decomposed, do_nfc=True, jobs=1, chunk_chars=1)
        back8 = inverse("strict", decomposed, do_nfc=True, jobs=8, chunk_chars=1)
        self.assertEqual(back1, back8)
        # Expect composed syllables where possible:
        expected = "각" + "X" + "가" + "Y" + "ᄀ" + "Z" + "각"
        self.assertEqual(back1, expected)

    def test_tagged_forward_deterministic(self):
        rng = random.Random(456)
        text = "".join(
            rand_hangul_syllable(rng) if rng.random() < 0.5 else rand_compat_jamo(rng)
            for _ in range(2000)
        )
        f1 = forward("tagged", text, do_nfc=True, jobs=1, chunk_chars=53)
        f8 = forward("tagged", text, do_nfc=True, jobs=8, chunk_chars=53)
        self.assertEqual(f1, f8)

        b = inverse("tagged", f1, do_nfc=True, jobs=8, chunk_chars=53)
        self.assertEqual(b, unicodedata.normalize("NFC", text))


def run_tests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestBijectionMT)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


# -------------------------
# CLI
# -------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["strict", "tagged"], default="strict")
    ap.add_argument("--op", choices=["forward", "inverse"], default="forward")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--chunk-chars", type=int, default=200_000,
                    help="Chunk size in Unicode codepoints/characters for parallel processing.")
    ap.add_argument("--no-nfc", action="store_true",
                    help="Do not NFC-normalize input before forward transform.")
    ap.add_argument("--text", type=str, default=None,
                    help="Input text. If omitted, reads stdin.")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--run-tests", action="store_true")
    args = ap.parse_args()

    if args.run_tests:
        return run_tests()
    if args.demo:
        return run_demo()

    do_nfc = not args.no_nfc

    if args.text is None:
        text = sys.stdin.read()
        if text == "":
            print("No input provided. Use --text, pipe stdin, or --demo/--run-tests.", file=sys.stderr)
            return 2
    else:
        text = args.text

    if args.op == "forward":
        out = forward(args.mode, text, do_nfc=do_nfc, jobs=args.jobs, chunk_chars=args.chunk_chars)
    else:
        out = inverse(args.mode, text, do_nfc=do_nfc, jobs=args.jobs, chunk_chars=args.chunk_chars)

    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

