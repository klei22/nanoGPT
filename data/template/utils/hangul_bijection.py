#!/usr/bin/env python3
"""
hangul_bijection.py

Create a *bijective* (reversible) “canonicalized” representation for an internet
corpus containing Hangul, and map back to the exact original.

We implement TWO reversible strategies:

A) STRICT (lossless, no conversion of compatibility Jamo)
   - Normalize to NFC (optional but recommended)
   - Decompose modern Hangul syllables (U+AC00..U+D7A3) into canonical Jamo L/V/T
     (U+1100.., U+1161.., U+11A8..)
   - Leave everything else unchanged (including compatibility jamo U+3130..U+318F,
     extended jamo, punctuation, emoji, Latin, etc.)
   - Inverse: deterministically recompose L+V(+T) sequences into syllables.

B) TAGGED NORMALIZED (still bijective, but “normalizes” compatibility Jamo)
   - Same as STRICT, BUT additionally maps Hangul Compatibility Jamo (U+3130..U+318F)
     into canonical Jamo using NFKD, *and tags* them so we can restore the original
     compatibility codepoints on the way back.
   - This gives you a more uniform internal alphabet without losing reversibility.

Important:
- NFC normalization changes the original byte-level representation (e.g., NFD vs NFC).
  If you need to reconstruct the corpus *exactly as-is*, set --no-nfc.
  Otherwise, NFC is a good “canonical input” for corpora.

Usage:
  python3 hangul_bijection.py --mode strict --text "가가 ㄱㅏ"
  python3 hangul_bijection.py --mode tagged --text "ㄱㅏ"
  echo "한글 ㄱㅏ test" | python3 hangul_bijection.py --mode tagged

  # Show built-in examples:
  python3 hangul_bijection.py --demo
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple, Optional

# --- Hangul composition/decomposition constants (Unicode / Windows-style) ---
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
L_END   = L_BASE + L_COUNT - 1

V_START = V_BASE
V_END   = V_BASE + V_COUNT - 1

T_START = T_BASE + 1
T_END   = T_BASE + T_COUNT - 1

# Hangul Compatibility Jamo block
COMPAT_START = 0x3130
COMPAT_END   = 0x318F

# Tag markers for TAGGED mode.
# Use Private Use Area (PUA) sentinels to avoid collisions with real corpus text.
# U+E000..U+F8FF is PUA in BMP. We'll use two chars as wrappers:
TAG_OPEN  = "\uE000"
TAG_CLOSE = "\uE001"
# Inside the tag, we store the original compatibility codepoint as 4 hex digits.
# Example: TAG_OPEN + "3131" + TAG_CLOSE represents original U+3131 (ㄱ)


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


def decompose_syllable(cp: int) -> List[int]:
    """
    Decompose one Hangul syllable into canonical Jamo [L, V] or [L, V, T].
    """
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
        T = T_BASE + t_index
        out.append(T)
    return out


def compose_from_jamo_stream(cps: List[int]) -> List[int]:
    """
    Deterministically compose L+V(+T) sequences into precomposed Hangul syllables.
    Any non-matching codepoints pass through unchanged.
    """
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
                T = cps[i + 2]
                t_index = T - T_BASE
                advance = 3

            s_index = (l_index * N_COUNT) + (v_index * T_COUNT) + t_index
            out.append(S_BASE + s_index)
            i += advance
            continue

        out.append(cp)
        i += 1

    return out


def to_codepoints(s: str) -> List[int]:
    return [ord(ch) for ch in s]


def from_codepoints(cps: List[int]) -> str:
    return "".join(chr(cp) for cp in cps)


def maybe_nfc(s: str, do_nfc: bool) -> str:
    return unicodedata.normalize("NFC", s) if do_nfc else s


# -------------------------
# Mode A: STRICT bijection
# -------------------------
def forward_strict(s: str, do_nfc: bool) -> str:
    s = maybe_nfc(s, do_nfc)
    out: List[int] = []
    for ch in s:
        out.extend(decompose_syllable(ord(ch)))
    return from_codepoints(out)


def inverse_strict(s: str) -> str:
    cps = to_codepoints(s)
    recomposed = compose_from_jamo_stream(cps)
    return from_codepoints(recomposed)


# --------------------------------
# Mode B: TAGGED NORMALIZED bijection
# --------------------------------
def compat_to_canonical_jamo(cp: int) -> Optional[List[int]]:
    """
    Convert a single compatibility jamo cp to canonical jamo sequence using NFKD.
    If NFKD doesn't change it into canonical jamo, return None.
    """
    ch = chr(cp)
    nfkd = unicodedata.normalize("NFKD", ch)
    cps = [ord(c) for c in nfkd]
    # Accept only if all cps are canonical jamo ranges (L/V/T), not other stuff.
    # Often a compat jamo becomes a single canonical jamo, but keep it general.
    for x in cps:
        if not (is_L_jamo(x) or is_V_jamo(x) or is_T_jamo(x)):
            return None
    return cps if cps else None


def tag_original_compat(cp: int) -> List[int]:
    """
    Emit TAG_OPEN + 4 hex digits + TAG_CLOSE, as codepoints, to record original compat jamo.
    """
    hex4 = f"{cp:04X}"
    tagged = TAG_OPEN + hex4 + TAG_CLOSE
    return [ord(c) for c in tagged]


def forward_tagged(s: str, do_nfc: bool) -> str:
    s = maybe_nfc(s, do_nfc)
    out: List[int] = []
    for ch in s:
        cp = ord(ch)

        # Decompose syllables (same as strict)
        if is_hangul_syllable(cp):
            out.extend(decompose_syllable(cp))
            continue

        # Normalize compatibility jamo, but tag for lossless recovery
        if is_compat_jamo(cp):
            canon = compat_to_canonical_jamo(cp)
            if canon is not None:
                out.extend(tag_original_compat(cp))
                out.extend(canon)
                continue
            # If we can't map safely, keep as-is (still bijective)
            out.append(cp)
            continue

        # Everything else unchanged
        out.append(cp)

    return from_codepoints(out)


def parse_tagged_stream(cps: List[int]) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Parse TAG_OPEN + 4 hex digits + TAG_CLOSE markers.
    Returns:
      - cps_out: cps with the tag markers removed (but NOT applying the restoration)
      - tags: list of (index_in_cps_out, original_compat_cp) that should be restored later

    The tag is associated with the *next* canonical jamo token(s) that follow in the stream.
    We record the position where restoration should happen (before those canonical tokens).
    """
    out: List[int] = []
    tags: List[Tuple[int, int]] = []
    i = 0
    n = len(cps)

    while i < n:
        if cps[i] == ord(TAG_OPEN) and i + 6 < n:
            # Expect 4 hex digits then TAG_CLOSE
            hex_digits = cps[i + 1:i + 5]
            close = cps[i + 5]
            if close == ord(TAG_CLOSE) and all(
                (48 <= d <= 57) or (65 <= d <= 70) or (97 <= d <= 102) for d in hex_digits
            ):
                hex_str = "".join(chr(d) for d in hex_digits)
                orig_cp = int(hex_str, 16)
                # restoration point = current length of out
                tags.append((len(out), orig_cp))
                i += 6  # consume open + 4 digits + close
                continue

        out.append(cps[i])
        i += 1

    return out, tags


def inverse_tagged(s: str) -> str:
    cps = to_codepoints(s)

    # 1) Remove tag markers but remember where to restore original compat cps
    cps_no_tags, tags = parse_tagged_stream(cps)

    # 2) First, compose syllables from L+V(+T) sequences
    composed = compose_from_jamo_stream(cps_no_tags)

    # 3) Now restore compatibility jamo at recorded positions:
    #    The forward_tagged inserted: [TAG][canonical jamo...]
    #    We restore the original compatibility cp by replacing the canonical jamo tokens
    #    that immediately follow the tag. But how many tokens?
    #
    # Deterministic rule:
    #   - If the next token after the tag (in cps_no_tags) is a canonical jamo (L/V/T),
    #     we replace exactly ONE canonical token with the original compatibility cp.
    #
    # This matches the most common mapping (compat -> single canonical jamo).
    # If you want multi-token compat mappings, we can extend this rule, but in practice
    # for U+3130..U+318F it's usually 1:1 for these letters.
    #
    # Since we already composed syllables, canonical jamo may have been consumed into a syllable,
    # so restoration must happen BEFORE composition if you want to exactly restore compat in-place.
    # Therefore: we should restore on cps_no_tags first, then compose. Let's do that properly:

    # Redo step 2 with correct ordering:
    cps_restored = list(cps_no_tags)
    # Apply tags in reverse order to keep indices valid
    for idx, orig_cp in reversed(tags):
        if idx < len(cps_restored) and (is_L_jamo(cps_restored[idx]) or is_V_jamo(cps_restored[idx]) or is_T_jamo(cps_restored[idx])):
            cps_restored[idx] = orig_cp
        else:
            # If unexpected structure, we just insert the original compat cp
            cps_restored.insert(idx, orig_cp)

    # Now compose syllables from any remaining canonical L/V/T sequences
    final = compose_from_jamo_stream(cps_restored)
    return from_codepoints(final)


# -------------------------
# Pretty printing / demo
# -------------------------
def show(label: str, s: str) -> None:
    def cps_str(x: str) -> str:
        return " ".join(f"U+{ord(c):04X}" for c in x)
    print(f"{label}:")
    print(f"  text: {s!r}")
    print(f"  cps : {cps_str(s)}")
    print()


@dataclass
class RoundTripResult:
    forward: str
    back: str
    ok: bool


def run_roundtrip(mode: str, text: str, do_nfc: bool) -> RoundTripResult:
    if mode == "strict":
        fwd = forward_strict(text, do_nfc)
        back = inverse_strict(fwd)
    elif mode == "tagged":
        fwd = forward_tagged(text, do_nfc)
        back = inverse_tagged(fwd)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return RoundTripResult(fwd, back, back == (unicodedata.normalize("NFC", text) if do_nfc else text))


def demo(do_nfc: bool) -> int:
    examples = [
        # Pure syllables
        "한글 테스트",
        # Mix of precomposed and canonical Jamo (NFD-like)
        "가가",  # '가' plus '가'
        # Compatibility Jamo examples
        "ㄱㅏ ㄴㅏ ㄷㅏ",  # common compat sequence
        # Mixed scripts, emoji, punctuation
        "K-pop 🐱 가요! ㄱㅏ? 가!",
        # Edge case: standalone canonical jamo
        "ᄀ ᅡ ᆨ",
    ]

    for ex in examples:
        print("=" * 80)
        print(f"INPUT (do_nfc={do_nfc}): {ex!r}\n")

        for mode in ("strict", "tagged"):
            r = run_roundtrip(mode, ex, do_nfc)
            print(f"MODE: {mode}")
            show("Original (effective input)", unicodedata.normalize("NFC", ex) if do_nfc else ex)
            show("Forward (bijective form)", r.forward)
            show("Inverse (restored)", r.back)
            print(f"Round-trip OK: {r.ok}\n")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["strict", "tagged"], default="strict",
                    help="strict: leave compat jamo unchanged; tagged: normalize compat jamo but tag for exact recovery.")
    ap.add_argument("--text", type=str, default=None,
                    help="Input text. If omitted, reads stdin.")
    ap.add_argument("--no-nfc", action="store_true",
                    help="Do NOT NFC-normalize the input first (preserves exact original normalization).")
    ap.add_argument("--demo", action="store_true",
                    help="Run built-in demonstrations with multiple examples.")
    args = ap.parse_args()

    do_nfc = not args.no_nfc

    if args.demo:
        return demo(do_nfc)

    if args.text is None:
        text = sys.stdin.read()
        if text == "":
            print("No input provided. Use --text, pipe stdin, or --demo.", file=sys.stderr)
            return 2
    else:
        text = args.text

    r = run_roundtrip(args.mode, text, do_nfc)
    print(f"MODE={args.mode} (do_nfc={do_nfc})")
    show("Original (effective input)", unicodedata.normalize("NFC", text) if do_nfc else text)
    show("Forward (bijective form)", r.forward)
    show("Inverse (restored)", r.back)
    print(f"Round-trip OK: {r.ok}")
    return 0 if r.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

