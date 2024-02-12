# DialogSum

This folder contains scripts compatible with the DialoguSum dataset.

## Script usage

### Run the get_dataset.py script to convert into json and txt formats

```bash
python get_dataset.py
```
The above script will create the following files:

```
input.txt
processed_combined_dialogsum.json
```

### Tokenization

The `input.txt` can be utilized directly with the `prepare.py` script, e.g.

```bash
python3 prepare.py input.txt --method tiktoken
```

And then one can utilize this folder for training.

## More information about DialogSum

### Dataset Description

The set consists of 13,460 dialogues (+100 for topic generation), spanning across
various real-life scenarios such as medical consultations, job interviews, and
daily conversations.

### Dataset Description

* Languages: English
* Number of Dialogues: 13,460 dialogues
* Data Splits:
    + Training: 12,460 dialogues
    + Validation: 500 dialogues
    + Testing: 1,500 dialogues
    + Holdout: 100 dialogues (Only includes id, dialogue, topic)

Each dialogue is accompanied by a human-written summary and topic, crafted based
on specific criteria to ensure they convey the most salient information in a
concise and formal manner.

### Data Fields

* `dialogue`: Text of the dialogue.
* `summary`: Human-written summary of the dialogue.
* `topic`: Human-written topic/one-liner of the dialogue.
* `id`: Unique identifier for each dialogue instance.

### Attribution and Citation

For more information, visit the following links from the [DIALOGSum GitHub repository](https://github.com/cylnlp/dialogsum)
or the [ACL Anthology page](https://aclanthology.org/2021.findings-acl.449/).

bibtex
```
@inproceedings{chen-etal-2021-dialogsum,
    title = "{D}ialog{S}um: {A} Real-Life Scenario Dialogue Summarization Dataset",
    author = "Chen, Yulong  and
      Liu, Yang  and
      Chen, Liang  and
      Zhang, Yue",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.449",
    doi = "10.18653/v1/2021.findings-acl.449",
    pages = "5062--5074",
}
```

