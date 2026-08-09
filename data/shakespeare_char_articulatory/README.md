# Shakespeare character articulatory multicontext

Builds aligned multicontext lanes from Tiny Shakespeare: lowercase text, case class (uppercase/lowercase/other), and approximate articulatory-phonetic lanes for letters with an `other` category for non-letters. The source text is Tiny Shakespeare from Andrej Karpathy's char-rnn examples.

Run:

```bash
data/shakespeare_char_articulatory/get_dataset.sh
```

The script writes one nanoGPT character dataset per lane under `data/shakespeare_char_articulatory/<lane>/` and a `manifest.json` listing the `--multicontext_datasets` arguments.
