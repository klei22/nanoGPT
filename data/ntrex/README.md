# ntrex dataset setup

This folder documents how to pull the multilingual `ntrex` parquet files convert
them into a tokenized dataset that matches the rest of the repository.

## 1) Gather the raw parquet files

Use the provided `get_dataset.sh` script to download the parquet shards and flatten the fields you care about into a single `input.txt` file. The script defaults to the dataset homepage URL on Hugging Face and uses the shared parquet helper from `data/template/utils`.

```bash
bash get_dataset.sh
```

The script currently pulls four language columns (`eng_Latn`, `kor_Hang`, `zho_Hans`, and `jpn_Jpan`) and prefixes each row with `#EN:\n` to keep splits visible in the final text file. Adjust the `include_keys`/`value_prefix` arrays in the script if you want to extract other languages from the parquet schema. The helper will download the parquet files, convert each to JSON, then emit the requested keys line-by-line into `input.txt`.

## 2) Tokenize the text

After `input.txt` is created, run the standard template tokenizer to build train/validation binaries. You can switch methods or vocab size depending on your needs.

```bash
python3 ../template/prepare.py -t input.txt --method sentencepiece --vocab_size 32000
```

This produces `train.bin`, `val.bin`, and `meta.pkl` in the current directory for use with `train.py` and related scripts. If you prefer TikToken or character-level tokenization, pass `--method tiktoken` or `--method char` instead.

## 3) Re-run after changes

If you tweak the include keys or add more parquet files, delete `input.txt` (and optionally the `downloaded_parquets` and `json_output` helper folders) before rerunning `get_dataset.sh` so the regenerated text reflects your changes.


# Original Dataset PAge

- [NTREX Main Page](https://huggingface.co/datasets/mteb/NTREX)

# How to cite original dataset

From the NTREX project's Huggingface page:

> If you use this dataset, please cite the dataset as well as mteb, as this
> dataset likely includes additional processing as a part of the MMTEB
> Contribution.

```
@inproceedings{federmann-etal-2022-ntrex,
  address = {Online},
  author = {Federmann, Christian and Kocmi, Tom and Xin, Ying},
  booktitle = {Proceedings of the First Workshop on Scaling Up Multilingual Evaluation},
  month = {nov},
  pages = {21--24},
  publisher = {Association for Computational Linguistics},
  title = {{NTREX}-128 {--} News Test References for {MT} Evaluation of 128 Languages},
  url = {https://aclanthology.org/2022.sumeval-1.4},
  year = {2022},
}


@article{enevoldsen2025mmtebmassivemultilingualtext,
  title={MMTEB: Massive Multilingual Text Embedding Benchmark},
  author={Kenneth Enevoldsen and Isaac Chung and Imene Kerboua and Márton Kardos and Ashwin Mathur and David Stap and Jay Gala and Wissam Siblini and Dominik Krzemiński and Genta Indra Winata and Saba Sturua and Saiteja Utpala and Mathieu Ciancone and Marion Schaeffer and Gabriel Sequeira and Diganta Misra and Shreeya Dhakal and Jonathan Rystrøm and Roman Solomatin and Ömer Çağatan and Akash Kundu and Martin Bernstorff and Shitao Xiao and Akshita Sukhlecha and Bhavish Pahwa and Rafał Poświata and Kranthi Kiran GV and Shawon Ashraf and Daniel Auras and Björn Plüster and Jan Philipp Harries and Loïc Magne and Isabelle Mohr and Mariya Hendriksen and Dawei Zhu and Hippolyte Gisserot-Boukhlef and Tom Aarsen and Jan Kostkan and Konrad Wojtasik and Taemin Lee and Marek Šuppa and Crystina Zhang and Roberta Rocca and Mohammed Hamdy and Andrianos Michail and John Yang and Manuel Faysse and Aleksei Vatolin and Nandan Thakur and Manan Dey and Dipam Vasani and Pranjal Chitale and Simone Tedeschi and Nguyen Tai and Artem Snegirev and Michael Günther and Mengzhou Xia and Weijia Shi and Xing Han Lù and Jordan Clive and Gayatri Krishnakumar and Anna Maksimova and Silvan Wehrli and Maria Tikhonova and Henil Panchal and Aleksandr Abramov and Malte Ostendorff and Zheng Liu and Simon Clematide and Lester James Miranda and Alena Fenogenova and Guangyu Song and Ruqiya Bin Safi and Wen-Ding Li and Alessia Borghini and Federico Cassano and Hongjin Su and Jimmy Lin and Howard Yen and Lasse Hansen and Sara Hooker and Chenghao Xiao and Vaibhav Adlakha and Orion Weller and Siva Reddy and Niklas Muennighoff},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2502.13595},
  year={2025},
  url={https://arxiv.org/abs/2502.13595},
  doi = {10.48550/arXiv.2502.13595},
}

@article{muennighoff2022mteb,
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Lo{\"\i}c and Reimers, Nils},
  title = {MTEB: Massive Text Embedding Benchmark},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2210.07316},
  year = {2022}
  url = {https://arxiv.org/abs/2210.07316},
  doi = {10.48550/ARXIV.2210.07316},
}
```

