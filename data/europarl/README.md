# Europarl (Helsinki-NLP/europarl)

This directory contains scripts compatible with the Europarl dataset.

## Summary
The OPUS Europarl dataset is a parallel corpus extracted from the European Parliament web site by Philipp Koehn (University of Edinburgh). It is commonly used for machine translation and cross-lingual word embedding research.

## Data source
- Dataset: https://huggingface.co/datasets/Helsinki-NLP/europarl
- OPUS Europarl: https://opus.nlpl.eu/Europarl/corpus/version/Europarl
- Additional info: http://www.statmt.org/europarl/

## Languages
Every pair of the following languages is available in the corpus:

bg, cs, da, de, el, en, es, et, fi, fr, hu, it, lt, lv, nl, pl, pt, ro, sk, sl, sv.

## Dataset structure
- **Rows:** 185,506,545
- **Auto-converted Parquet size:** 35.6 GB
- **Data split:** train only

## Usage
From this directory, run:

```bash
./get_dataset.sh
```

This uses `data/template/utils/get_parquet_translation_dataset.py` to download the Parquet shards for the `en-pt` subset and emit `input.txt` with English and Portuguese entries.

## Licensing
The dataset comes with the same license as the original sources. See the OPUS Europarl page for details: https://opus.nlpl.eu/Europarl/corpus/version/Europarl

## Citation
Please cite the following works when using this corpus:

```bibtex
@inproceedings{koehn-2005-europarl,
    title = "{E}uroparl: A Parallel Corpus for Statistical Machine Translation",
    author = "Koehn, Philipp",
    booktitle = "Proceedings of Machine Translation Summit X: Papers",
    month = sep,
    year = "2005",
    address = "Phuket, Thailand",
    url = "https://aclanthology.org/2005.mtsummit-papers.11",
    pages = "79--86",
}
```

```bibtex
@inproceedings{tiedemann-2012-parallel,
    title = "Parallel Data, Tools and Interfaces in {OPUS}",
    author = {Tiedemann, J{"o}rg},
    editor = "Calzolari, Nicoletta  and
      Choukri, Khalid  and
      Declerck, Thierry  and
      Do{\u{g}}an, Mehmet U{\u{g}}ur  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Moreno, Asuncion  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Eighth International Conference on Language Resources and Evaluation ({LREC}'12)",
    month = may,
    year = "2012",
    address = "Istanbul, Turkey",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2012/pdf/463_Paper.pdf",
    pages = "2214--2218",
}
```

