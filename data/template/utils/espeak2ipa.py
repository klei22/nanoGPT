#!/usr/bin/env python3
# espeak2ipa.py
#
# Generic IPA transcription using espeak-ng for ANY supported voice.
# Defaults to "shan" (you can override with --lang).
#
# Features (modeled after your en2ipa.py):
# - JSON list mode (--mode json): in-place update of a JSON list file
# - Text mode (--mode text): line-by-line transcription to output file (or overwrite input)
# - Optional wrapping for untranscribed/unparseable tokens: [[[[[...]]]]]
# - Multithreading with ordered output
# - Rich progress bar
# - Byte coverage stats (based on ORIGINAL tokens; wrapper overhead excluded)
#
# Notes:
# - "transcribed_bytes" counts bytes of ORIGINAL tokens we ATTEMPT to send to espeak
#   (tokens that contain at least one Unicode letter). Digits/punct count as not_transcribed.
# - espeak-ng voices vary; if a voice is unavailable, you'll get an error/empty output.

import subprocess
import argparse
import re
import json
from typing import List, Tuple, Optional, Dict, Any
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading


WRAP_PREFIX = "[[[[["
WRAP_SUFFIX = "]]]]]"
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

counter_unparseable = 0
counter_lock = threading.Lock()


def utf8_len(s: str) -> int:
    return len(s.encode("utf-8"))


def token_has_letter(tok: str) -> bool:
    # "letter" across scripts (Latin, Han, Kana, Arabic, etc.)
    return any(ch.isalpha() for ch in tok)


def transcribe_espeak(token: str, lang: str, wrapper: bool = False) -> str:
    """
    Transcribe a token via espeak-ng.
    If transcription fails (empty output / exception), return wrapped or original token.
    """
    global counter_unparseable
    try:
        result = subprocess.run(
            ["espeak-ng", "-q", "-v", lang, "--ipa", token],
            capture_output=True,
            text=True
        )
        out = (result.stdout or "").strip().replace("ㆍ", " ")
        if not out:
            if wrapper:
                return f"{WRAP_PREFIX}{token}{WRAP_SUFFIX}"
            with counter_lock:
                counter_unparseable += 1
            return token
        return out
    except Exception:
        if wrapper:
            return f"{WRAP_PREFIX}{token}{WRAP_SUFFIX}"
        with counter_lock:
            counter_unparseable += 1
        return token


def handle_token(tok: str, lang: str, wrapper: bool) -> str:
    """
    Decide whether to transcribe:
      - digits -> passthrough
      - tokens with any letter -> transcribe via espeak
      - otherwise (punct/symbol) -> passthrough
    """
    if tok.isdigit():
        return tok
    if token_has_letter(tok):
        return transcribe_espeak(tok, lang=lang, wrapper=wrapper)
    return tok


def tokens_to_ipa_string(tokens: List[str], lang: str, wrapper: bool) -> str:
    out: List[str] = []
    for tok in tokens:
        if re.match(r"\w+", tok):
            out.append(handle_token(tok, lang=lang, wrapper=wrapper))
        else:
            out.append(tok)
    return " ".join(out)


def _worker_sentence(sentence: str, lang: str, wrapper: bool, stats: Optional[Dict[str, int]] = None) -> str:
    """
    Tokenize & transcribe one sentence/line.
    If stats is provided, updates byte counts based on ORIGINAL tokens:
      - transcribed_bytes: tokens containing at least one letter
      - not_transcribed_bytes: digits + punctuation/symbols + other \w tokens with no letters
    """
    tokens = _WORD_RE.findall(sentence)

    if stats is not None:
        for tok in tokens:
            b = utf8_len(tok)
            if re.match(r"\w+", tok):
                if tok.isdigit():
                    stats["not_transcribed_bytes"] += b
                elif token_has_letter(tok):
                    stats["transcribed_bytes"] += b
                else:
                    stats["not_transcribed_bytes"] += b
            else:
                stats["not_transcribed_bytes"] += b

    return tokens_to_ipa_string(tokens, lang=lang, wrapper=wrapper)


def _progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    )


def transcribe_json_list(
    json_text_or_obj,
    input_json_key: str,
    output_json_key: str,
    lang: str,
    wrapper: bool,
    multithread: bool,
    workers: int,
    stats: Optional[Dict[str, int]] = None,
) -> Optional[str]:
    """
    JSON list mode: reads a JSON list of objects, writes output_json_key for each object.
    Returns JSON string (pretty printed).
    """
    try:
        data = json.loads(json_text_or_obj) if isinstance(json_text_or_obj, str) else json_text_or_obj
        if not isinstance(data, list):
            raise ValueError("JSON data should be a list of objects.")
    except Exception as e:
        print(f"Error: {e}")
        return None

    n = len(data)
    if stats is None:
        stats = {"transcribed_bytes": 0, "not_transcribed_bytes": 0}
    else:
        stats.setdefault("transcribed_bytes", 0)
        stats.setdefault("not_transcribed_bytes", 0)

    if n == 0:
        return json.dumps(data, ensure_ascii=False, indent=4)

    if not multithread or workers <= 1:
        with _progress() as progress:
            task = progress.add_task("Processing JSON items", total=n)
            for item in data:
                if input_json_key in item:
                    sentence = item[input_json_key]
                    item[output_json_key] = _worker_sentence(sentence, lang=lang, wrapper=wrapper, stats=stats)
                progress.update(task, advance=1)
    else:
        # ordered results
        results: List[Tuple[int, str]] = [None] * n  # type: ignore
        per_item_stats: List[Dict[str, int]] = [None] * n  # type: ignore

        jobs = [(i, data[i].get(input_json_key, "")) for i in range(n)]

        with _progress() as progress:
            task = progress.add_task(f"Processing JSON items (mt x{workers})", total=n)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                future_to_idx = {}
                for idx, sentence in jobs:
                    local_stats = {"transcribed_bytes": 0, "not_transcribed_bytes": 0}
                    per_item_stats[idx] = local_stats
                    fut = ex.submit(_worker_sentence, sentence, lang, wrapper, local_stats)
                    future_to_idx[fut] = idx

                for fut in as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    try:
                        res = fut.result()
                    except Exception as e:
                        res = f"Error: {e}"
                    results[idx] = (idx, res)
                    progress.update(task, advance=1)

        # merge stats
        for st in per_item_stats:
            stats["transcribed_bytes"] += st.get("transcribed_bytes", 0)
            stats["not_transcribed_bytes"] += st.get("not_transcribed_bytes", 0)

        # write back in original order
        for idx, item in enumerate(data):
            if input_json_key in item:
                item[output_json_key] = results[idx][1]

    return json.dumps(data, ensure_ascii=False, indent=4)


def transcribe_text_lines(
    lines: List[str],
    lang: str,
    wrapper: bool,
    multithread: bool,
    workers: int,
    stats: Optional[Dict[str, int]] = None,
) -> List[str]:
    """
    Text mode: input is one sentence per line. Output is one IPA line per input line.
    """
    n = len(lines)
    if stats is None:
        stats = {"transcribed_bytes": 0, "not_transcribed_bytes": 0}
    else:
        stats.setdefault("transcribed_bytes", 0)
        stats.setdefault("not_transcribed_bytes", 0)

    if n == 0:
        return []

    if not multithread or workers <= 1:
        out_lines: List[str] = []
        with _progress() as progress:
            task = progress.add_task("Processing text lines", total=n)
            for line in lines:
                raw = line.rstrip("\n")
                out_lines.append(_worker_sentence(raw, lang=lang, wrapper=wrapper, stats=stats))
                progress.update(task, advance=1)
        return out_lines

    out_lines: List[str] = [None] * n  # type: ignore
    per_item_stats: List[Dict[str, int]] = [None] * n  # type: ignore

    with _progress() as progress:
        task = progress.add_task(f"Processing text lines (mt x{workers})", total=n)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_to_idx = {}
            for i in range(n):
                local_stats = {"transcribed_bytes": 0, "not_transcribed_bytes": 0}
                per_item_stats[i] = local_stats
                fut = ex.submit(_worker_sentence, lines[i].rstrip("\n"), lang, wrapper, local_stats)
                future_to_idx[fut] = i

            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                try:
                    out_lines[i] = fut.result()
                except Exception as e:
                    out_lines[i] = f"Error: {e}"
                progress.update(task, advance=1)

    for st in per_item_stats:
        stats["transcribed_bytes"] += st.get("transcribed_bytes", 0)
        stats["not_transcribed_bytes"] += st.get("not_transcribed_bytes", 0)

    return out_lines


def finalize_and_print_stats(stats: Dict[str, int], stats_json_path: Optional[str] = None) -> Dict[str, Any]:
    transcribed = int(stats.get("transcribed_bytes", 0))
    not_tx = int(stats.get("not_transcribed_bytes", 0))
    total = transcribed + not_tx
    pct_tx = (transcribed / total * 100.0) if total else 0.0
    pct_not = (not_tx / total * 100.0) if total else 0.0

    out_stats: Dict[str, Any] = {
        "transcribed_bytes": transcribed,
        "not_transcribed_bytes": not_tx,
        "total_bytes": total,
        "pct_transcribed": pct_tx,
        "pct_not_transcribed": pct_not,
        "unparseable_tokens": counter_unparseable,
    }

    print("\n=== Byte Coverage Stats (based on ORIGINAL tokens) ===")
    print(f"Transcribed bytes      : {out_stats['transcribed_bytes']}")
    print(f"Not transcribed bytes  : {out_stats['not_transcribed_bytes']}")
    print(f"Total bytes (counted)  : {out_stats['total_bytes']}")
    print(f"% transcribed          : {out_stats['pct_transcribed']:.2f}%")
    print(f"% not transcribed      : {out_stats['pct_not_transcribed']:.2f}%")
    print(f"Unparseable tokens     : {out_stats['unparseable_tokens']}")

    if stats_json_path:
        with open(stats_json_path, "w", encoding="utf-8") as sf:
            json.dump(out_stats, sf, ensure_ascii=False, indent=2)
        print(f"Stats JSON written to: {stats_json_path}")

    return out_stats


def main():
    parser = argparse.ArgumentParser(
        description="Generic IPA transcription using espeak-ng for any supported voice (default: shan). "
                    "Supports JSON list mode and plain-text line mode, with byte coverage stats."
    )
    parser.add_argument("input_file", type=str, help="Path to the input file (JSON list or plain text).")

    # Language / voice
    parser.add_argument("--lang", default="shan",
                        help="espeak-ng voice/language code (default: shan). Example: en, fr, de, es, ja, zh, etc.")

    # Mode selection
    parser.add_argument("--mode", choices=["json", "text"], default="json",
                        help='Processing mode. "json" expects a JSON list; "text" treats file as plain text.')

    # JSON mode params
    parser.add_argument("--input_json_key", type=str,
                        help="JSON key to read sentences from (required for --mode json).")
    parser.add_argument("--output_json_key", type=str, default="ipa",
                        help='JSON key to store IPA (default: "ipa").')

    # Text mode params
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file path for text mode. Defaults to overwriting input.")

    # Wrapper option
    parser.add_argument("--wrapper", default=False, action=argparse.BooleanOptionalAction,
                        help="Wrap unparseable tokens with [[[[[...]]]]] (default: false).")

    # Multithreading options
    parser.add_argument("--multithread", default=False, action=argparse.BooleanOptionalAction,
                        help="Enable multithreading while preserving output order.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                        help="Number of worker threads when --multithread is enabled (default: CPU count).")

    # Stats output
    parser.add_argument("--stats_json", type=str, default=None,
                        help="Optional: write byte coverage stats as JSON to this path (in addition to printing).")

    args = parser.parse_args()

    # clamp workers
    if args.workers is None or args.workers < 1:
        args.workers = 1

    stats = {"transcribed_bytes": 0, "not_transcribed_bytes": 0}

    try:
        if args.mode == "json":
            if not args.input_json_key:
                raise ValueError("--input_json_key is required when --mode json")

            with open(args.input_file, "r", encoding="utf-8") as f:
                input_content = f.read()

            updated_json = transcribe_json_list(
                input_content,
                input_json_key=args.input_json_key,
                output_json_key=args.output_json_key,
                lang=args.lang,
                wrapper=args.wrapper,
                multithread=args.multithread,
                workers=args.workers,
                stats=stats,
            )
            if updated_json is not None:
                # matches your existing style: overwrite JSON input file
                with open(args.input_file, "w", encoding="utf-8") as f:
                    f.write(updated_json)
                print(f"✅ Successfully updated JSON data in '{args.input_file}'")

        else:
            with open(args.input_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            out_lines = transcribe_text_lines(
                lines,
                lang=args.lang,
                wrapper=args.wrapper,
                multithread=args.multithread,
                workers=args.workers,
                stats=stats,
            )

            target_path = args.output_file if args.output_file else args.input_file
            with open(target_path, "w", encoding="utf-8") as f:
                f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
            print(f"✅ Successfully wrote transcribed text to '{target_path}'")

        finalize_and_print_stats(stats, stats_json_path=args.stats_json)

    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
    except ValueError as ve:
        print(f"Error: {ve}")


if __name__ == "__main__":
    main()

