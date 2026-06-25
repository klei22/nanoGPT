#!/usr/bin/env python3
"""Fast bijection checks for the simplified_hanzi_mc lane contract."""
from __future__ import annotations

from build_simplified_hanzi_mc import (
    LANES,
    NON_HANZI,
    PLACEHOLDER,
    TRADITIONAL_ONLY,
    CJK_RANGES,
    encode_char,
    encode_non_hanzi_value,
    is_simplified_hanzi,
)
from decode_multicontext_sample import decode_non_hanzi_value


def decode_rows(rows: list[dict[str, str]]) -> str:
    decoded: list[str] = []
    for row in rows:
        if row["char"] == NON_HANZI:
            decoded.append(decode_non_hanzi_value(encode_non_hanzi_value(row["non_hanzi"])))
        else:
            decoded.append(row["char"])
    return "".join(decoded)


def assert_full_text_round_trip() -> None:
    text = (
        "#Title:\n女冠子·绿云高髻\n#Poem:\n"
        "绿云高髻，点翠匀红时世。\\\t🙂\r\n"
        "传统fixture:體龍門馬愛學國風書樂車東長萬與興貓鳥魚\n"
        "Supplementary CJK:𠀀𫝀𰀀"
    )
    rows = [encode_char(ch) for ch in text]
    assert decode_rows(rows) == text


def assert_cjk_ranges_preserve_char_lane() -> None:
    checked = 0
    for start, end in CJK_RANGES:
        for codepoint in range(start, end + 1):
            ch = chr(codepoint)
            row = encode_char(ch)
            if ch in TRADITIONAL_ONLY:
                assert row["char"] == NON_HANZI
                assert row["non_hanzi"] == ch
            else:
                assert is_simplified_hanzi(ch)
                assert row["char"] == ch
                assert row["non_hanzi"] == PLACEHOLDER
            assert set(row) == set(LANES)
            checked += 1
    print(f"checked_cjk_codepoints={checked}")


def assert_non_hanzi_scalar_encoding() -> None:
    for ch in ["#", "\n", "\r", "\t", "\\", "🙂", "。", "\u2028", "\u2029"]:
        encoded = encode_non_hanzi_value(ch)
        assert "\n" not in encoded and "\r" not in encoded
        assert decode_non_hanzi_value(encoded) == ch


def main() -> None:
    assert_full_text_round_trip()
    assert_cjk_ranges_preserve_char_lane()
    assert_non_hanzi_scalar_encoding()
    print("simplified_hanzi_mc_bijection_ok")


if __name__ == "__main__":
    main()
