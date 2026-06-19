# Group Theory Dataset

This dataset generator produces sequences of valid group operations for
basic algebraic structures. It can be used to train models to predict the
next element of a group given a previous element and an operator.

Currently two group families are supported:

- **Cyclic groups** of arbitrary order with optional wrap around (group closure).
- **Dihedral groups** with optional restriction to their rotational
  subgroups.

The scripts allow custom symbols for the group elements and the operators.
This enables curriculum experiments where the same algebraic rules are
presented using different character sets.

If the requested group order requires more symbols than provided via
`--state-symbols`, additional characters are generated automatically in
ascending Unicode order starting from `!`. Characters used for the
operators are skipped to avoid collisions.

## Usage

To create a text dataset:

```bash
python generate_dataset.py --group cyclic --order 3 \
    --length 10 --state-symbols abc --operator-symbol + \
    --output dataset.txt
```

Use `--no-closure` to disable wrap around for cyclic groups.
For dihedral groups use `--group dihedral --order <n>`.

A simple validator is included and can be invoked with

```bash
python generate_dataset.py --validate dataset.txt
```

which will parse the file and check that every operation is correct.
