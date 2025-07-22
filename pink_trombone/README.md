# Pink Trombone Integration

This folder demonstrates how to generate control parameters for the [Pink Trombone](../data/pink_trombone/pink-trombone-master) speech synthesizer using `espeak-ng` to obtain IPA transcriptions.

The workflow is:

1. **IPA Conversion** – Convert English text to IPA with `espeak-ng`.
2. **Phoneme Mapping** – Map each IPA symbol to a set of Pink Trombone parameters.
3. **Parameter Emission** – Emit the numeric parameter frames that can be consumed by the Pink Trombone library.

The `ipa_to_params.py` script shows a minimal example of this process. It is intended as a starting point for building a comprehensive dataset that covers the full IPA inventory.

## Requirements

- Python 3.8+
- `espeak-ng` installed and accessible on the command line

## Example Usage

```bash
python ipa_to_params.py "hello world"
```

The script prints a JSON array of parameter frames corresponding to the IPA output for the given text. Each frame is a dictionary whose keys match the control parameters exposed by the Rust implementation.

Extend the `PHONEME_MAP` dictionary in `ipa_to_params.py` to cover additional IPA symbols and adjust parameter values to your needs.
