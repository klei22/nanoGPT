## Annotated Hanzi Dataset (OPUS-100 + hanzipy)

This dataset extends the OPUS-100 English ↔ Mandarin Chinese parallel corpus with
per-character Hanzi annotations powered by the `hanzipy` library. Each Mandarin
entry is enriched with definitions, pinyin, frequency data, decomposition
information, phonetic regularity, and radical meanings when available.

The folder started as a copy of the existing OPUS-100 utilities so we can reuse
its download + preprocessing workflow and then layer the Hanzi annotations on
its JSON output.

## Create `annotated_hanzi` from OPUS-100

If you need to re-create this dataset folder from scratch, you can start by
copying the OPUS-100 utilities:

```bash
cp -R data/opus-100 data/annotated_hanzi
```

## Requirements

Install hanzipy before running the annotation step:

```bash
pip install hanzipy
```

## Workflow

### 1) Download OPUS-100 English ↔ Mandarin data

From this directory, download the English to Mandarin Chinese data and emit
JSON:

```bash
python3 get_dataset.py -f en -t zh
```

This creates Parquet downloads in `downloaded_parquets/` and JSON files in
`json_output/`.

### 2) Annotate Mandarin with Hanzi metadata

Annotate a single JSON file:

```bash
python3 annotate_hanzi.py \
  --input_json json_output/opus-100-train.json \
  --output_json annotated_json_output/opus-100-train.json \
  --zh_key zh
```

Annotate every JSON file in a directory:

```bash
python3 annotate_hanzi.py --input_dir json_output --output_dir annotated_json_output --zh_key zh
```

### Output schema

Each record gains a new `annotated_hanzi` field:

```json
{
  "translation": {
    "en": "Well, the fire department has its own investigative unit.",
    "zh": "消防部门有自己的调查单位。"
  },
  "annotated_hanzi": {
    "characters": ["消", "防", "部", "门", "有", "自", "己", "的", "调", "查", "单", "位"],
    "by_character": {
      "消": {
        "definitions": [{"traditional": "消", "simplified": "消", "pinyin": "xiao1", "definition": "to disappear"}],
        "pinyin": ["xiao1"],
        "frequency": {"number": 712, "character": "消", "count": "58452", "percentage": "69.633"},
        "decomposition": {"character": "消", "once": ["氵", "肖"], "radical": ["氵", "肖"], "graphical": ["氵", "肖"]},
        "phonetic_regularity": {"xiao1": {"character": "消", "component": ["肖"], "phonetic_pinyin": ["xiao1"], "regularity": [1]}},
        "radical_meanings": {"氵": "water"}
      }
    }
  }
}
```

## Notes

- The annotation step caches per-character lookups so repeated Hanzi across the
  dataset are only queried once.
- Adjust `--zh_key` if your OPUS-100 JSON uses a different language key.

## Sources

- OPUS-100: https://huggingface.co/datasets/Helsinki-NLP/opus-100
- Hanzipy: https://pypi.org/project/hanzipy/
