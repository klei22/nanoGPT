#!/usr/bin/env python3
# data/template/utils/en2ipa.py

import subprocess
import argparse
import re
import json
from typing import List, Tuple, Optional, Dict, Any
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading

counter = 0
counter_lock = threading.Lock()

WRAP_PREFIX = "[[[[["
WRAP_SUFFIX = "]]]]]"

_WORD_RE = re.compile(r'\w+|[^\w\s]', re.UNICODE)


def utf8_len(s: str) -> int:
    return len(s.encode("utf-8"))


def is_english_token(tok: str) -> bool:
    # Matches your original intent: “contains any a-z letter”
    return any('a' <= ch.lower() <= 'z' for ch in tok)


def transcribe_english(sentence, wrapper=False):
    """Transcribe an English sentence into its phonemes using espeak."""
    try:
        result = subprocess.run(
            ["espeak-ng", "-q", "-v", "en", "--ipa", sentence],
            capture_output=True,
            text=True
        )
        transcription = result.stdout.strip().replace("ㆍ", " ")
        if "(en)" in transcription:
            return f"{WRAP_PREFIX}{sentence}{WRAP_SUFFIX}" if wrapper else sentence
        return transcription
    except Exception as e:
        return f"Error in transcribing English: {str(e)}"


def handle_mixed_language(word, wrapper=False):
    """Handle a word with potential English, Language, or number content."""
    global counter
    if word.isdigit():
        return word
    elif is_english_token(word):
        return transcribe_english(word, wrapper=wrapper)
    else:
        if wrapper:
            return f"{WRAP_PREFIX}{word}{WRAP_SUFFIX}"
        else:
            # thread-safe increment (your existing stat)
            with counter_lock:
                counter += 1
            return word


def transcribe_tokens_to_string(tokens: List[str], wrapper: bool) -> str:
    result = []
    for tok in tokens:
        if re.match(r'\w+', tok):
            result.append(handle_mixed_language(tok, wrapper=wrapper))
        else:
            result.append(tok)
    return " ".join(result)


def _worker_sentence(sentence: str, wrapper: bool, stats: Optional[Dict[str, int]] = None) -> str:
    """
    Worker function: tokenize and transcribe one sentence/line.
    If stats is provided, updates:
      - transcribed_bytes: UTF-8 bytes of ORIGINAL tokens that were transcribed (English tokens)
      - not_transcribed_bytes: UTF-8 bytes of ORIGINAL tokens not transcribed (digits, punctuation, non-English words)
    Counts are based on ORIGINAL tokens, so wrapper overhead is excluded automatically.
    """
    tokens = _WORD_RE.findall(sentence)

    if stats is not None:
        for tok in tokens:
            b = utf8_len(tok)
            if re.match(r'\w+', tok):
                if tok.isdigit():
                    stats["not_transcribed_bytes"] += b
                elif is_english_token(tok):
                    stats["transcribed_bytes"] += b
                else:
                    stats["not_transcribed_bytes"] += b
            else:
                stats["not_transcribed_bytes"] += b

    return transcribe_tokens_to_string(tokens, wrapper=wrapper)


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


def transcribe_multilingual(
    sentences,
    input_json_key=None,
    output_json_key='ipa',
    wrapper=False,
    multithread: bool = False,
    workers: int = 0,
    stats: Optional[Dict[str, int]] = None,
):
    """Transcribe multilingual sentences (JSON list mode)."""
    try:
        data = json.loads(sentences) if isinstance(sentences, str) else sentences
        if not isinstance(data, list):
            raise ValueError("JSON data should be a list of objects.")

        n = len(data)
        if n == 0:
            return json.dumps(data, ensure_ascii=False, indent=4)

        if stats is None:
            stats = {"transcribed_bytes": 0, "not_transcribed_bytes": 0}
        else:
            stats.setdefault("transcribed_bytes", 0)
            stats.setdefault("not_transcribed_bytes", 0)

        if not multithread or workers <= 1:
            # Single-threaded path (original behavior)
            with _progress() as progress:
                task = progress.add_task("Processing JSON items", total=n)
                for item in data:
                    if input_json_key in item:
                        sentence = item[input_json_key]
                        item[output_json_key] = _worker_sentence(sentence, wrapper, stats=stats)
                    progress.update(task, advance=1)
        else:
            # Multithreaded path with ordered assembly
            results: List[Tuple[int, str]] = [None] * n  # type: ignore

            # prepare jobs
            jobs = []
            for idx, item in enumerate(data):
                sentence = item.get(input_json_key, "")
                jobs.append((idx, sentence))

            # Per-thread stats to avoid locks in hot path; merge at end
            per_thread_stats: List[Dict[str, int]] = []

            def submit_job(ex, idx_sentence):
                idx, sentence = idx_sentence
                local_stats = {"transcribed_bytes": 0, "not_transcribed_bytes": 0}
                per_thread_stats.append(local_stats)
                return ex.submit(_worker_sentence, sentence, wrapper, local_stats), idx

            with _progress() as progress:
                task = progress.add_task(f"Processing JSON items (mt x{workers})", total=n)
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    future_to_idx = {}
                    for idx_sentence in jobs:
                        fut, idx = submit_job(ex, idx_sentence)
                        future_to_idx[fut] = idx

                    for fut in as_completed(future_to_idx):
                        idx = future_to_idx[fut]
                        try:
                            res = fut.result()
                        except Exception as e:
                            res = f"Error: {e}"
                        results[idx] = (idx, res)
                        progress.update(task, advance=1)

            # merge per-thread stats
            for st in per_thread_stats:
                stats["transcribed_bytes"] += st.get("transcribed_bytes", 0)
                stats["not_transcribed_bytes"] += st.get("not_transcribed_bytes", 0)

            # write back in original order
            for idx, item in enumerate(data):
                if input_json_key in item:
                    item[output_json_key] = results[idx][1]

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error: {e}")
        return None

    return json.dumps(data, ensure_ascii=False, indent=4)


def transcribe_text_lines(
    lines: List[str],
    wrapper: bool,
    multithread: bool = False,
    workers: int = 0,
    stats: Optional[Dict[str, int]] = None,
) -> List[str]:
    """Transcribe a plain-text file line-by-line."""
    n = len(lines)
    if n == 0:
        return []

    if stats is None:
        stats = {"transcribed_bytes": 0, "not_transcribed_bytes": 0}
    else:
        stats.setdefault("transcribed_bytes", 0)
        stats.setdefault("not_transcribed_bytes", 0)

    if not multithread or workers <= 1:
        out_lines: List[str] = []
        with _progress() as progress:
            task = progress.add_task("Processing text lines", total=n)
            for line in lines:
                raw = line.rstrip("\n")
                out_lines.append(_worker_sentence(raw, wrapper, stats=stats))
                progress.update(task, advance=1)
        return out_lines
    else:
        out_lines: List[str] = [None] * n  # type: ignore

        # Per-thread stats (avoid global lock)
        per_thread_stats: List[Dict[str, int]] = [None] * n  # type: ignore

        with _progress() as progress:
            task = progress.add_task(f"Processing text lines (mt x{workers})", total=n)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                future_to_idx = {}
                for i in range(n):
                    local_stats = {"transcribed_bytes": 0, "not_transcribed_bytes": 0}
                    per_thread_stats[i] = local_stats
                    fut = ex.submit(_worker_sentence, lines[i].rstrip("\n"), wrapper, local_stats)
                    future_to_idx[fut] = i

                for fut in as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    try:
                        out_lines[idx] = fut.result()
                    except Exception as e:
                        out_lines[idx] = f"Error: {e}"
                    progress.update(task, advance=1)

        # merge stats
        for st in per_thread_stats:
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
    }

    print("\n=== Byte Coverage Stats (based on ORIGINAL tokens) ===")
    print(f"Transcribed bytes      : {out_stats['transcribed_bytes']}")
    print(f"Not transcribed bytes  : {out_stats['not_transcribed_bytes']}")
    print(f"Total bytes (counted)  : {out_stats['total_bytes']}")
    print(f"% transcribed          : {out_stats['pct_transcribed']:.2f}%")
    print(f"% not transcribed      : {out_stats['pct_not_transcribed']:.2f}%")

    if stats_json_path:
        with open(stats_json_path, "w", encoding="utf-8") as sf:
            json.dump(out_stats, sf, ensure_ascii=False, indent=2)
        print(f"Stats JSON written to: {stats_json_path}")

    return out_stats


def main():
    parser = argparse.ArgumentParser(
        description='Transcribe multilingual content into IPA phonemes. Supports JSON list mode and plain-text line mode.'
    )
    parser.add_argument('input_file', type=str, help='Path to the input file (JSON list or plain text).')

    # Mode selection
    parser.add_argument('--mode', choices=['json', 'text'], default='json',
                        help='Processing mode. "json" expects a JSON list; "text" treats file as plain text.')

    # JSON mode params
    parser.add_argument('--input_json_key', type=str, help='JSON key to read sentences from (required for --mode json).')
    parser.add_argument('--output_json_key', type=str, default='ipa', help='JSON key to store IPA (default: "ipa").')

    # Text mode params
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file path for text mode. Defaults to overwriting input.')

    # Common options
    parser.add_argument("--wrapper", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="Wrap unparseable non-English text with [[[[[...]]]]] for later recovery.")

    # Multithreading options
    parser.add_argument("--multithread", default=False, action=argparse.BooleanOptionalAction,
                        help="Enable multithreading for faster processing while preserving output order.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                        help="Number of worker threads when --multithread is enabled (default: CPU count).")

    # NEW: stats output
    parser.add_argument("--stats_json", type=str, default=None,
                        help="Optional: write byte coverage stats as JSON to this path (in addition to printing).")

    args = parser.parse_args()

    # clamp workers
    if args.workers is None or args.workers < 1:
        args.workers = 1

    stats = {"transcribed_bytes": 0, "not_transcribed_bytes": 0}

    try:
        if args.mode == 'json':
            if not args.input_json_key:
                raise ValueError("--input_json_key is required when --mode json")
            with open(args.input_file, 'r', encoding='utf-8') as f:
                input_content = f.read()
            updated_json_data = transcribe_multilingual(
                input_content,
                args.input_json_key,
                args.output_json_key,
                wrapper=args.wrapper,
                multithread=args.multithread,
                workers=args.workers,
                stats=stats,
            )
            if updated_json_data:
                with open(args.input_file, 'w', encoding='utf-8') as f:
                    f.write(updated_json_data)
                print(f"✅ Successfully updated JSON data in '{args.input_file}'")
        else:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            out_lines = transcribe_text_lines(
                lines,
                wrapper=args.wrapper,
                multithread=args.multithread,
                workers=args.workers,
                stats=stats,
            )
            target_path = args.output_file if args.output_file else args.input_file
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
            print(f"✅ Successfully wrote transcribed text to '{target_path}'")

        finalize_and_print_stats(stats, stats_json_path=args.stats_json)

        print(f"📊 Stats: {counter} unparseable words (only counted when --no-wrapper)")

    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
    except ValueError as ve:
        print(f"Error: {ve}")


if __name__ == '__main__':
    main()

