# Pink Trombone Dataset

This folder provides a starting point for generating numeric
parameters for the Rust based `pink-trombone` speech synthesizer.
The goal is to convert plain English text into a sequence of
numbers that describe vocal tract and glottal parameters.
These numbers can then be fed into the `pink-trombone` crate to
produce sound.

The workflow is:

1. **Convert text to IPA** using `espeak`.
   ```bash
   espeak --ipa -q "hello world"
   ```
   yields something like `həlˈəʊ wˈɜːld`.
2. **Map each IPA symbol to Pink Trombone parameters.**
   This repository includes a simple Python example (see
   `ipa_to_pink.py`) which demonstrates how to translate an IPA string
   into numeric settings such as `tongue_index`, `tongue_diameter`,
   `target_frequency` and other controls exposed by the
   `PinkTrombone` struct.
3. **Emit numeric sequences.**  The example script prints a JSON list
   describing the parameters over time.  These can be consumed by a
   Rust or Python program that drives the synthesizer.

### Simplified Phoneme Mapping

The example provides a very small mapping for demonstration
purposes only.  Real speech will require a comprehensive table
covering all IPA symbols of interest.  Each entry specifies the
values used to call the setter methods on `PinkTrombone`.  For
instance:

```python
PHONEME_TO_PARAMS = {
    "a": {"tongue_index": 16.0, "tongue_diameter": 2.8, "target_frequency": 140},
    "i": {"tongue_index": 20.0, "tongue_diameter": 1.6, "target_frequency": 220},
    "u": {"tongue_index": 12.0, "tongue_diameter": 2.4, "target_frequency": 110},
    # ... add additional phonemes here ...
}
```

### Generating Parameter Sequences

Run the script with some text.  It will call `espeak`, parse the IPA
string and output a JSON list of parameter dictionaries.

```bash
python3 ipa_to_pink.py "hello world" > params.json
```

The resulting file can be loaded and fed frame by frame into the
synthesizer.  See the example program in
`pink-trombone-master/examples/pink-trombone.rs` for how the Rust
API expects parameter updates.

### Notes

- The mapping provided here is intentionally simple to illustrate the
  concept.  Creating natural speech requires tuning many more
  parameters and transitions between them.
- `espeak` must be installed and accessible on your system.
- Feel free to extend the mapping dictionary or add a more advanced
  dataset format.

