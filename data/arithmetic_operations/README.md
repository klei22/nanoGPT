# Arithmetic Operations Dataset

This folder contains utilities for building synthetic datasets of chained arithmetic
operations along with a helper script for validating model outputs.

## Dataset generation

The `generate_dataset.py` script creates chains of operations that are composed from
addition, multiplication, barrel shifts, and digit reversal. Operations are applied
sequentially with each new value feeding into the next step. Two textual layouts are
supported:

* **stacked** – matches the multi-line format shown below where values and operations
  alternate:

  ```
  0
  +1
  1
  *102
  102
  R
  210
  ```

* **inline** – emits expressions in the form `a+b=c` or `aR=c` on individual lines.

Run `python generate_dataset.py --help` to view all options. A few notable flags are:

* `--operations` – choose a comma-separated subset of operations from
  `addition,multiplication,shift_right,shift_left,reverse`.
* `--modulo` – apply a modulo after every step (default `1000`, use `-1` to disable).
* `--format` – select either `stacked` or `inline` output.
* `--num-sequences` / `--num-steps` – control dataset size.

Example:

```bash
python generate_dataset.py --num-sequences 4 --num-steps 8 --format inline --output data.txt
```

## Output evaluation

`evaluate_operations.py` inspects a text file (for example, `sample-XXX.txt` produced by
`train.py`) and reports how many lines conform to the expected format. Both stacked and
inline styles are supported.

```bash
python evaluate_operations.py --input sample.txt
```

Use `--format` to override the auto-detected layout and `--include-empty` if blank lines
should be counted as incorrect rows.
