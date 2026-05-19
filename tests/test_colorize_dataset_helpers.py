import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colorize_dataset import build_colorized_token_rows


def test_build_colorized_token_rows_shape_and_fields():
    ids = [1, 2, 3]
    scalars = [0.2, 0.8, 0.5]

    def decode_fn(values):
        return f"T{values[0]}"

    rows = build_colorized_token_rows(ids, scalars, decode_fn)

    assert len(rows) == 3
    assert rows[0]["token_id"] == 1
    assert rows[0]["token"] == "T1"
    assert abs(rows[1]["scalar"] - 0.8) < 1e-8
    for row in rows:
        assert 0.0 <= row["heat"] <= 1.0
