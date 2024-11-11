# GlotCC-V1.0 Korean Subset

This repository provides a script compatible with the Korean subset (`kor-Hang`)
of **GlotCC-V1.0**, a document-level dataset derived from CommonCrawl. The
dataset supports multilingual and minority language research, and the pipeline
is open-source.

## Overview

**GlotCC-V1.0** is a broad-coverage dataset covering over 1000 languages,
designed for natural language processing tasks. The Korean subset (`kor-Hang`)
includes document-level data in Hangul script.

### How to Use

The script provided downloads and processes the Korean subset of GlotCC-V1.0
using the following command:

```bash
#!/bin/bash

url="https://huggingface.co/datasets/cis-lmu/GlotCC-V1/tree/main/v1.0/kor-Hang"

python3 ./utils/get_parquet_dataset.py \
  --url "${url}" \
  --include_keys "content" \
  --value_prefix $'\n'
```

### License

The dataset is licensed as follows:
- **Data**: Licensed under the [CommonCrawl Terms of Use](https://commoncrawl.org/terms-of-use/).
- **Packaging, Metadata, and Annotations**: Released under the **Creative Commons CC0 license**.

### Citation

If you find **GlotCC-V1.0** useful in your research, please cite it as follows:

```bibtex
@article{kargaran2024glotcc,
  title     = {Glot{CC}: An Open Broad-Coverage CommonCrawl Corpus and Pipeline for Minority Languages},
  author    = {Kargaran, Amir Hossein and Yvon, Fran{\c{c}}ois and Sch{\"u}tze, Hinrich},
  journal   = {Advances in Neural Information Processing Systems},
  year      = {2024},
  url       = {https://arxiv.org/abs/2410.23825}
}
```

For more details on the dataset, visit the [Hugging Face page](https://huggingface.co/datasets/cis-lmu/GlotCC-V1).
