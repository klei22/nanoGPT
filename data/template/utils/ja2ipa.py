#!/usr/bin/env python3
# ja2ipa_io.py
import argparse
import json
import sys
from collections import OrderedDict
from typing import Tuple, Optional, List, Dict, Any

from tqdm import tqdm
import pykakasi.kakasi as kakasi

# 1) Attempt optional imports
try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# ========== Kakasi Converter Setup ==========
kks = kakasi()
kks.setMode('J', 'H')  # Kanji -> Hiragana
kks.setMode('H', 'H')  # Hiragana -> Hiragana
kks.setMode('K', 'H')  # Katakana -> Hiragana
conv = kks.getConverter()

kana_mapper = OrderedDict([
    ("ゔぁ","bˈa"),
    ("ゔぃ","bˈi"),
    ("ゔぇ","bˈe"),
    ("ゔぉ","bˈo"),
    ("ゔゃ","bˈʲa"),
    ("ゔゅ","bˈʲɯ"),
    ("ゔゃ","bˈʲa"),
    ("ゔょ","bˈʲo"),

    ("ゔ","bˈɯ"),

    ("あぁ","aː"),
    ("いぃ","iː"),
    ("いぇ","je"),
    ("いゃ","ja"),
    ("うぅ","ɯː"),
    ("えぇ","eː"),
    ("おぉ","oː"),
    ("かぁ","kˈaː"),
    ("きぃ","kˈiː"),
    ("くぅ","kˈɯː"),
    ("くゃ","kˈa"),
    ("くゅ","kˈʲɯ"),
    ("くょ","kˈʲo"),
    ("けぇ","kˈeː"),
    ("こぉ","kˈoː"),
    ("がぁ","gˈaː"),
    ("ぎぃ","gˈiː"),
    ("ぐぅ","gˈɯː"),
    ("ぐゃ","gˈʲa"),
    ("ぐゅ","gˈʲɯ"),
    ("ぐょ","gˈʲo"),
    ("げぇ","gˈeː"),
    ("ごぉ","gˈoː"),
    ("さぁ","sˈaː"),
    ("しぃ","ɕˈiː"),
    ("すぅ","sˈɯː"),
    ("すゃ","sˈʲa"),
    ("すゅ","sˈʲɯ"),
    ("すょ","sˈʲo"),
    ("せぇ","sˈeː"),
    ("そぉ","sˈoː"),
    ("ざぁ","zˈaː"),
    ("じぃ","dʑˈiː"),
    ("ずぅ","zˈɯː"),
    ("ずゃ","zˈʲa"),
    ("ずゅ","zˈʲɯ"),
    ("ずょ","zˈʲo"),
    ("ぜぇ","zˈeː"),
    ("ぞぉ","zˈeː"),
    ("たぁ","tˈaː"),
    ("ちぃ","tɕˈiː"),
    ("つぁ","tsˈa"),
    ("つぃ","tsˈi"),
    ("つぅ","tsˈɯː"),
    ("つゃ","tɕˈa"),
    ("つゅ","tɕˈɯ"),
    ("つょ","tɕˈo"),
    ("つぇ","tsˈe"),
    ("つぉ","tsˈo"),
    ("てぇ","tˈeː"),
    ("とぉ","tˈoː"),
    ("だぁ","dˈaː"),
    ("ぢぃ","dʑˈiː"),
    ("づぅ","dˈɯː"),
    ("づゃ","zˈʲa"),
    ("づゅ","zˈʲɯ"),
    ("づょ","zˈʲo"),
    ("でぇ","dˈeː"),
    ("どぉ","dˈoː"),
    ("なぁ","nˈaː"),
    ("にぃ","nˈiː"),
    ("ぬぅ","nˈɯː"),
    ("ぬゃ","nˈʲa"),
    ("ぬゅ","nˈʲɯ"),
    ("ぬょ","nˈʲo"),
    ("ねぇ","nˈeː"),
    ("のぉ","nˈoː"),
    ("はぁ","hˈaː"),
    ("ひぃ","çˈiː"),
    ("ふぅ","ɸˈɯː"),
    ("ふゃ","ɸˈʲa"),
    ("ふゅ","ɸˈʲɯ"),
    ("ふょ","ɸˈʲo"),
    ("へぇ","hˈeː"),
    ("ほぉ","hˈoː"),
    ("ばぁ","bˈaː"),
    ("びぃ","bˈiː"),
    ("ぶぅ","bˈɯː"),
    ("ふゃ","ɸˈʲa"),
    ("ぶゅ","bˈʲɯ"),
    ("ふょ","ɸˈʲo"),
    ("べぇ","bˈeː"),
    ("ぼぉ","bˈoː"),
    ("ぱぁ","pˈaː"),
    ("ぴぃ","pˈiː"),
    ("ぷぅ","pˈɯː"),
    ("ぷゃ","pˈʲa"),
    ("ぷゅ","pˈʲɯ"),
    ("ぷょ","pˈʲo"),
    ("ぺぇ","pˈeː"),
    ("ぽぉ","pˈoː"),
    ("まぁ","mˈaː"),
    ("みぃ","mˈiː"),
    ("むぅ","mˈɯː"),
    ("むゃ","mˈʲa"),
    ("むゅ","mˈʲɯ"),
    ("むょ","mˈʲo"),
    ("めぇ","mˈeː"),
    ("もぉ","mˈoː"),
    ("やぁ","jˈaː"),
    ("ゆぅ","jˈɯː"),
    ("ゆゃ","jˈaː"),
    ("ゆゅ","jˈɯː"),
    ("ゆょ","jˈoː"),
    ("よぉ","jˈoː"),
    ("らぁ","ɽˈaː"),
    ("りぃ","ɽˈiː"),
    ("るぅ","ɽˈɯː"),
    ("るゃ","ɽˈʲa"),
    ("るゅ","ɽˈʲɯ"),
    ("るょ","ɽˈʲo"),
    ("れぇ","ɽˈeː"),
    ("ろぉ","ɽˈoː"),
    ("わぁ","ɯˈaː"),
    ("をぉ","oː"),

    ("う゛","bˈɯ"),
    ("でぃ","dˈi"),
    ("でぇ","dˈeː"),
    ("でゃ","dˈʲa"),
    ("でゅ","dˈʲɯ"),
    ("でょ","dˈʲo"),
    ("てぃ","tˈi"),
    ("てぇ","tˈeː"),
    ("てゃ","tˈʲa"),
    ("てゅ","tˈʲɯ"),
    ("てょ","tˈʲo"),
    ("すぃ","sˈi"),
    ("ずぁ","zˈɯa"),
    ("ずぃ","zˈi"),
    ("ずぅ","zˈɯ"),
    ("ずゃ","zˈʲa"),
    ("ずゅ","zˈʲɯ"),
    ("ずょ","zˈʲo"),
    ("ずぇ","zˈe"),
    ("ずぉ","zˈo"),
    ("きゃ","kˈʲa"),
    ("きゅ","kˈʲɯ"),
    ("きょ","kˈʲo"),
    ("しゃ","ɕˈʲa"),
    ("しゅ","ɕˈʲɯ"),
    ("しぇ","ɕˈʲe"),
    ("しょ","ɕˈʲo"),
    ("ちゃ","tɕˈa"),
    ("ちゅ","tɕˈɯ"),
    ("ちぇ","tɕˈe"),
    ("ちょ","tɕˈo"),
    ("とぅ","tˈɯ"),
    ("とゃ","tˈʲa"),
    ("とゅ","tˈʲɯ"),
    ("とょ","tˈʲo"),
    ("どぁ","dˈoa"),
    ("どぅ","dˈɯ"),
    ("どゃ","dˈʲa"),
    ("どゅ","dˈʲɯ"),
    ("どょ","dˈʲo"),
    ("どぉ","dˈoː"),
    ("にゃ","nˈʲa"),
    ("にゅ","nˈʲɯ"),
    ("にょ","nˈʲo"),
    ("ひゃ","çˈʲa"),
    ("ひゅ","çˈʲɯ"),
    ("ひょ","çˈʲo"),
    ("みゃ","mˈʲa"),
    ("みゅ","mˈʲɯ"),
    ("みょ","mˈʲo"),
    ("りゃ","ɽˈʲa"),
    ("りぇ","ɽˈʲe"),
    ("りゅ","ɽˈʲɯ"),
    ("りょ","ɽˈʲo"),
    ("ぎゃ","gˈʲa"),
    ("ぎゅ","gˈʲɯ"),
    ("ぎょ","gˈʲo"),
    ("ぢぇ","dʑˈe"),
    ("ぢゃ","dʑˈa"),
    ("ぢゅ","dʑˈɯ"),
    ("ぢょ","dʑˈo"),
    ("じぇ","dʑˈe"),
    ("じゃ","dʑˈa"),
    ("じゅ","dʑˈɯ"),
    ("じょ","dʑˈo"),
    ("びゃ","bˈʲa"),
    ("びゅ","bˈʲɯ"),
    ("びょ","bˈʲo"),
    ("ぴゃ","pˈʲa"),
    ("ぴゅ","pˈʲɯ"),
    ("ぴょ","pˈʲo"),
    ("うぁ","ɯˈa"),
    ("うぃ","ɯˈi"),
    ("うぇ","ɯˈe"),
    ("うぉ","ɯˈo"),
    ("うゃ","ɯˈʲa"),
    ("うゅ","ɯˈʲɯ"),
    ("うょ","ɯˈʲo"),
    ("ふぁ","ɸˈa"),
    ("ふぃ","ɸˈi"),
    ("ふぅ","ɸˈɯ"),
    ("ふゃ","ɸˈʲa"),
    ("ふゅ","ɸˈʲɯ"),
    ("ふょ","ɸˈʲo"),
    ("ふぇ","ɸˈe"),
    ("ふぉ","ɸˈo"),

    ("あ","a"),
    ("い","i"),
    ("う","ɯ"),
    ("え","e"),
    ("お","o"),
    ("か","kˈa"),
    ("き","kˈi"),
    ("く","kˈɯ"),
    ("け","kˈe"),
    ("こ","kˈo"),
    ("さ","sˈa"),
    ("し","ɕˈi"),
    ("す","sˈɯ"),
    ("せ","sˈe"),
    ("そ","sˈo"),
    ("た","tˈa"),
    ("ち","tɕˈi"),
    ("つ","tsˈɯ"),
    ("て","tˈe"),
    ("と","tˈo"),
    ("な","nˈa"),
    ("に","nˈi"),
    ("ぬ","nˈɯ"),
    ("ね","nˈe"),
    ("の","nˈo"),
    ("は","hˈa"),
    ("ひ","çˈi"),
    ("ふ","ɸˈɯ"),
    ("へ","hˈe"),
    ("ほ","hˈo"),
    ("ま","mˈa"),
    ("み","mˈi"),
    ("む","mˈɯ"),
    ("め","mˈe"),
    ("も","mˈo"),
    ("ら","ɽˈa"),
    ("り","ɽˈi"),
    ("る","ɽˈɯ"),
    ("れ","ɽˈe"),
    ("ろ","ɽˈo"),
    ("が","gˈa"),
    ("ぎ","gˈi"),
    ("ぐ","gˈɯ"),
    ("げ","gˈe"),
    ("ご","gˈo"),
    ("ざ","zˈa"),
    ("じ","dʑˈi"),
    ("ず","zˈɯ"),
    ("ぜ","zˈe"),
    ("ぞ","zˈo"),
    ("だ","dˈa"),
    ("ぢ","dʑˈi"),
    ("づ","zˈɯ"),
    ("で","dˈe"),
    ("ど","dˈo"),
    ("ば","bˈa"),
    ("び","bˈi"),
    ("ぶ","bˈɯ"),
    ("べ","bˈe"),
    ("ぼ","bˈo"),
    ("ぱ","pˈa"),
    ("ぴ","pˈi"),
    ("ぷ","pˈɯ"),
    ("ぺ","pˈe"),
    ("ぽ","pˈo"),
    ("や","jˈa"),
    ("ゆ","jˈɯ"),
    ("よ","jˈo"),
    ("わ","ɯˈa"),
    ("ゐ","i"),
    ("ゑ","e"),
    ("ん","ɴ"),
    ("っ","ʔ"),
    ("ー","ː"),

    ("ぁ","a"),
    ("ぃ","i"),
    ("ぅ","ɯ"),
    ("ぇ","e"),
    ("ぉ","o"),
    ("ゎ","ɯˈa"),
    ("ぉ","o"),

    ("を","o")
])

nasal_sound = OrderedDict([
    # before m, p, b
    ("ɴm","mm"),
    ("ɴb","mb"),
    ("ɴp","mp"),

    # before k, g
    ("ɴk","ŋk"),
    ("ɴg","ŋg"),

    # before t, d, n, s, z, ɽ
    ("ɴt","nt"),
    ("ɴd","nd"),
    ("ɴn","nn"),
    ("ɴs","ns"),
    ("ɴz","nz"),
    ("ɴɽ","nɽ"),

    ("ɴɲ","ɲɲ"),
])

# ========== Basic Conversions ==========
def to_hiragana(text: str) -> str:
    """Convert JP text to Hiragana via Kakasi."""
    return conv.do(text)

def hiragana_to_ipa(text: str) -> str:
    """Convert Hiragana to IPA using kana_mapper + nasal_sound."""
    for k, v in kana_mapper.items():
        text = text.replace(k, v)
    for k, v in nasal_sound.items():
        text = text.replace(k, v)
    return text


# ========== 2) MeCab Morphological Tokenization ==========
def mecab_spaced_reading(text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    if not MECAB_AVAILABLE:
        return None, None, None, None

    tagger = MeCab.Tagger()
    node = tagger.parseToNode(text)

    tokens_original = []
    tokens_for_hira = []
    pos_tokens = []

    while node:
        surface = node.surface
        features = node.feature.split(",")
        if len(features) >= 1:
            pos = features[0]
            tokens_original.append(surface)
            pos_tokens.append(pos)
            if pos == "助詞" and surface == "は":
                tokens_for_hira.append("わ")
            else:
                tokens_for_hira.append(surface)
        else:
            tokens_original.append(surface)
            tokens_for_hira.append(surface)
            pos_tokens.append("UNK")
        node = node.next

    spaced_original = " ".join(tokens_original)
    spaced_for_hira = " ".join(tokens_for_hira)
    pos_tags = " ".join(pos_tokens)
    spaced_hira_subbed = to_hiragana(spaced_for_hira)
    spaced_hira_original = to_hiragana(spaced_original)

    return spaced_original, spaced_hira_subbed, spaced_hira_original, pos_tags


# ========== 3) spaCy Morphological Tokenization ==========
_spacy_nlp = None
def load_spacy_japanese():
    global _spacy_nlp
    if _spacy_nlp is None:
        _spacy_nlp = spacy.load("ja_core_news_sm")
    return _spacy_nlp

def spacy_spaced_reading(text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    if not SPACY_AVAILABLE:
        return None, None, None, None

    nlp = load_spacy_japanese()
    doc = nlp(text)

    tokens_original = []
    tokens_for_hira = []
    pos_tokens = []

    for token in doc:
        tokens_original.append(token.text)
        pos_tokens.append(token.pos_)
        if token.text == "は" and token.pos_ == "ADP":
            tokens_for_hira.append("わ")
        else:
            tokens_for_hira.append(token.text)

    spaced_original = " ".join(tokens_original)
    spaced_for_hira = " ".join(tokens_for_hira)
    pos_tags = " ".join(pos_tokens)
    spaced_hira_subbed = to_hiragana(spaced_for_hira)
    spaced_hira_original = to_hiragana(spaced_original)

    return spaced_original, spaced_hira_subbed, spaced_hira_original, pos_tags


# ========== 4) Unified "get spaced reading" function ==========
def get_spaced_reading(text: str, method: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    if method == "mecab":
        return mecab_spaced_reading(text)
    elif method == "spacy":
        return spacy_spaced_reading(text)
    else:
        return None, None, None, None


# ========== 5) Output Writers ==========
def write_json_array(output_file: str, out_array: List[Dict[str, Any]]) -> None:
    with open(output_file, "w", encoding="utf-8") as fout:
        json.dump(out_array, fout, ensure_ascii=False, indent=4)

def write_text_output(
    output_file: str,
    out_array: List[Dict[str, Any]],
    field: str = "spaced_ipa",
    include_sentence: bool = True,
    sep: str = "\t"
) -> None:
    with open(output_file, "w", encoding="utf-8") as fout:
        for obj in out_array:
            sent = obj.get("sentence", "")
            val = obj.get(field, "")
            if include_sentence:
                fout.write(f"{sent}{sep}{val}\n")
            else:
                fout.write(f"{val}\n")


def utf8_len(s: str) -> int:
    return len(s.encode("utf-8"))


# ========== 6) Main Processing Logic ==========
def process_japanese_text(
    input_file: str,
    output_file: str,
    json_inplace_update: bool = False,
    use_mecab: bool = False,
    use_spacy: bool = False,
    output_text: bool = False,
    text_field: str = "spaced_ipa",
    text_include_sentence: bool = True,
    text_sep: str = "\t",
    stats_json: Optional[str] = None,
):
    """
    Same behavior as before, plus byte coverage stats:
      - transcribed_bytes: UTF-8 bytes of ORIGINAL tokens that are considered "transcribed"
        (anything containing Japanese script: Hiragana/Katakana/Kanji)
      - not_transcribed_bytes: UTF-8 bytes of ORIGINAL tokens not transcribed (Latin, digits, punctuation)

    Counts are based on ORIGINAL tokens (wrapper overhead doesn't exist in this script).
    """
    if use_mecab and use_spacy:
        print("Error: Please choose either MeCab or spaCy, not both.")
        sys.exit(1)
    elif use_mecab:
        morph_method = "mecab"
    elif use_spacy:
        morph_method = "spacy"
    else:
        morph_method = None

    stats = {"transcribed_bytes": 0, "not_transcribed_bytes": 0}
    out_array: List[Dict[str, Any]] = []

    def is_japanese_char(ch: str) -> bool:
        o = ord(ch)
        # Hiragana, Katakana, CJK Unified Ideographs (basic), plus common punctuation blocks are excluded intentionally.
        return (0x3040 <= o <= 0x309F) or (0x30A0 <= o <= 0x30FF) or (0x4E00 <= o <= 0x9FFF)

    def is_japanese_token(tok: str) -> bool:
        return any(is_japanese_char(ch) for ch in tok)

    def count_sentence_bytes(sentence: str) -> None:
        # Tokenize similarly to your KR/ZH scripts for consistent accounting
        toks = re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)
        for tok in toks:
            b = utf8_len(tok)
            if re.match(r"\w+", tok):
                if tok.isdigit():
                    stats["not_transcribed_bytes"] += b
                elif is_japanese_token(tok):
                    stats["transcribed_bytes"] += b
                else:
                    stats["not_transcribed_bytes"] += b
            else:
                stats["not_transcribed_bytes"] += b

    if json_inplace_update:
        try:
            with open(input_file, "r", encoding="utf-8") as fin:
                data = json.load(fin)

            for entry in tqdm(data, desc="Processing JSON entries"):
                if "sentence" not in entry:
                    continue

                original_text = entry["sentence"]
                count_sentence_bytes(original_text)

                hira_unspaced = to_hiragana(original_text)
                ipa_unspaced = hiragana_to_ipa(hira_unspaced)

                out_obj: Dict[str, Any] = {
                    "sentence": original_text,
                    "unspaced_ipa": ipa_unspaced,
                    "spaced_original": "",
                    "spaced_hira_subbed": "",
                    "pos_tags": "",
                    "spaced_ipa": ""
                }

                if morph_method is not None:
                    spaced_original, spaced_hira_subbed, _, pos_tags = get_spaced_reading(original_text, morph_method)
                    out_obj["spaced_original"] = spaced_original if spaced_original is not None else ""
                    out_obj["spaced_hira_subbed"] = spaced_hira_subbed if spaced_hira_subbed is not None else ""
                    out_obj["pos_tags"] = pos_tags if pos_tags is not None else ""
                    out_obj["spaced_ipa"] = hiragana_to_ipa(out_obj["spaced_hira_subbed"]) if out_obj["spaced_hira_subbed"] else ""

                out_array.append(out_obj)

        except FileNotFoundError:
            print(f"Error: Input file '{input_file}' not found.")
            return
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{input_file}'.")
            return
        except Exception as e:
            print(f"An error occurred: {e}")
            return

    else:
        try:
            with open(input_file, "r", encoding="utf-8") as fin:
                lines = fin.readlines()

            for line in tqdm(lines, desc="Processing lines"):
                line = line.strip()
                if not line:
                    continue

                original_text = line
                count_sentence_bytes(original_text)

                hira_unspaced = to_hiragana(original_text)
                ipa_unspaced = hiragana_to_ipa(hira_unspaced)

                out_obj: Dict[str, Any] = {
                    "sentence": original_text,
                    "unspaced_ipa": ipa_unspaced,
                    "spaced_original": "",
                    "spaced_hira_subbed": "",
                    "pos_tags": "",
                    "spaced_ipa": ""
                }

                if morph_method is not None:
                    spaced_original, spaced_hira_subbed, _, pos_tags = get_spaced_reading(original_text, morph_method)
                    out_obj["spaced_original"] = spaced_original if spaced_original is not None else ""
                    out_obj["spaced_hira_subbed"] = spaced_hira_subbed if spaced_hira_subbed is not None else ""
                    out_obj["pos_tags"] = pos_tags if pos_tags is not None else ""
                    out_obj["spaced_ipa"] = hiragana_to_ipa(out_obj["spaced_hira_subbed"]) if out_obj["spaced_hira_subbed"] else ""

                out_array.append(out_obj)

        except FileNotFoundError:
            print(f"Error: Input file '{input_file}' not found.")
            return
        except Exception as e:
            print(f"An error occurred: {e}")
            return

    # OUTPUT (unchanged)
    if output_text:
        write_text_output(
            output_file=output_file,
            out_array=out_array,
            field=text_field,
            include_sentence=text_include_sentence,
            sep=text_sep
        )
    else:
        write_json_array(output_file=output_file, out_array=out_array)

    # Print + optional write stats
    transcribed = int(stats["transcribed_bytes"])
    not_tx = int(stats["not_transcribed_bytes"])
    total = transcribed + not_tx
    pct_tx = (transcribed / total * 100.0) if total else 0.0
    pct_not = (not_tx / total * 100.0) if total else 0.0

    out_stats = {
        "transcribed_bytes": transcribed,
        "not_transcribed_bytes": not_tx,
        "total_bytes": total,
        "pct_transcribed": pct_tx,
        "pct_not_transcribed": pct_not,
    }

    print("\n=== Byte Coverage Stats (based on ORIGINAL tokens) ===")
    print(f"Transcribed bytes      : {out_stats['transcribed_bytes']}")
    print(f"Not transcribed bytes  : {out_stats['not_transcribed_bytes']}")
    print(f"Total bytes (counted)  : {out_stats['total_bytes']}")
    print(f"% transcribed          : {out_stats['pct_transcribed']:.2f}%")
    print(f"% not transcribed      : {out_stats['pct_not_transcribed']:.2f}%")

    if stats_json:
        with open(stats_json, "w", encoding="utf-8") as sf:
            json.dump(out_stats, sf, ensure_ascii=False, indent=2)
        print(f"Stats JSON written to: {stats_json}")


# ========== 7) Command-Line Entry Point ==========
if __name__ == "__main__":
    import re  # local import to avoid changing your top imports too much

    parser = argparse.ArgumentParser(
        description=(
            "Convert JP text to IPA with optional morphological spacing and POS tagging.\n"
            "DEFAULT behavior matches original: input may be JSON (-j) or plain text, output is JSON array.\n"
            "You can output plain text with --text_output.\n"
            "NEW: prints byte coverage stats and can write them with --stats_json."
        )
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="input.txt",
        help="Path to the input file (JSON array with 'sentence' fields if -j, otherwise plain text)."
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default="output.json",
        help="Path to the output file (JSON by default, or text with --text_output)."
    )

    parser.add_argument(
        "-j", "--json_inplace_update",
        action="store_true",
        help="Treat input file as JSON array and update each entry."
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use_mecab", action="store_true",
                       help="Use MeCab for morphological tokenization (and forcing 'は' => 'わ').")
    group.add_argument("--use_spacy", action="store_true",
                       help="Use spaCy for morphological tokenization (and forcing 'は' => 'わ').")

    parser.add_argument("--text_output", action="store_true",
                        help="Write a plain-text output file (one line per sentence) instead of JSON.")
    parser.add_argument("--text_field", default="spaced_ipa",
                        help="Which field to emit in --text_output mode (default: spaced_ipa). "
                             "Common: unspaced_ipa, spaced_ipa, spaced_hira_subbed, pos_tags.")
    parser.add_argument("--text_no_sentence", action="store_true",
                        help="In --text_output mode, emit only the selected field (omit the original sentence).")
    parser.add_argument("--text_sep", default="\t",
                        help="Separator used between sentence and field in --text_output mode (default: tab).")

    parser.add_argument("--stats_json", type=str, default=None,
                        help="Optional: write byte coverage stats as JSON to this path (in addition to printing).")

    args = parser.parse_args()

    process_japanese_text(
        input_file=args.input_file,
        output_file=args.output_file,
        json_inplace_update=args.json_inplace_update,
        use_mecab=args.use_mecab,
        use_spacy=args.use_spacy,
        output_text=args.text_output,
        text_field=args.text_field,
        text_include_sentence=(not args.text_no_sentence),
        text_sep=args.text_sep,
        stats_json=args.stats_json,
    )

