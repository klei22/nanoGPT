#!/usr/bin/env python3
"""Download Dukascopy 1-minute FX candles (M1) for many pairs.

The downloader writes one gzipped CSV per instrument/day with columns:
timestamp_utc, open, high, low, close, volume.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import lzma
import os
import re
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Set, Tuple


DUKASCOPY_DATAFEED_BASE = "https://datafeed.dukascopy.com/datafeed"
DEFAULT_MAJORS: List[str] = [
    "eurusd",
    "gbpusd",
    "usdjpy",
    "usdchf",
    "usdcad",
    "audusd",
    "nzdusd",
]
DUKASCOPY_NODE_FX_CROSSES = "https://www.dukascopy-node.app/instruments/fx_crosses"
CANDLE_STRUCT_FMT = ">5if"
CANDLE_STRUCT_SIZE = struct.calcsize(CANDLE_STRUCT_FMT)


class RateLimiter:
    """Simple global rate limiter shared across worker threads."""

    def __init__(self, rps: float):
        self.rps = float(rps)
        self.min_interval = 1.0 / self.rps if self.rps > 0 else 0.0
        self._lock = threading.Lock()
        self._next_time = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                time.sleep(self._next_time - now)
                now = time.monotonic()
            self._next_time = now + self.min_interval


_thread_local = threading.local()


def _get_session(user_agent: str) -> Any:
    import requests

    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update({"User-Agent": user_agent, "Accept": "*/*"})
        _thread_local.session = sess
    return sess


def parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def daterange(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    d = start
    while d < end:
        yield d
        d += dt.timedelta(days=1)


def load_pairs_from_file(path: str) -> List[str]:
    pairs: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip().lower()
            if not line:
                continue
            if not re.fullmatch(r"[a-z0-9]{6}", line):
                raise ValueError(f"Invalid instrument id in {path}: {line!r} (expected 6 chars like 'eurusd')")
            pairs.add(line)
    return sorted(pairs)


def fetch_instrument_ids_from_dukascopy_node(url: str, timeout_s: int, user_agent: str) -> List[str]:
    import requests

    r = requests.get(url, timeout=timeout_s, headers={"User-Agent": user_agent})
    r.raise_for_status()
    html = r.text
    candidates: Set[str] = set()
    for pattern in (r'href="/instrument/([A-Za-z0-9]{6})"', r"<code[^>]*>\s*([A-Za-z0-9]{6})\s*</code>", r"`\s*([A-Za-z0-9]{6})\s*`"):
        for match in re.findall(pattern, html, flags=re.I):
            candidates.add(match.lower())
    return sorted(candidates)


def guess_tick_scale(instrument_id: str, default_non_jpy: int = 5, jpy_scale: int = 3) -> int:
    return jpy_scale if len(instrument_id) == 6 and instrument_id.lower().endswith("jpy") else default_non_jpy


def build_candle_url(instrument_id: str, day: dt.date, side: str) -> str:
    return (
        f"{DUKASCOPY_DATAFEED_BASE}/{instrument_id.upper()}/"
        f"{day.year}/{day.month - 1:02d}/{day.day:02d}/"
        f"{side.upper()}_candles_min_1.bi5"
    )


def decode_bi5_candles(bi5_bytes: bytes, day: dt.date, tick_scale: int) -> List[Tuple[str, str, str, str, str, str]]:
    decompressed = lzma.decompress(bi5_bytes)
    midnight = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc)
    price_div = 10**tick_scale
    price_fmt = f"{{:.{tick_scale}f}}"
    rows: List[Tuple[str, str, str, str, str, str]] = []
    usable = len(decompressed) - (len(decompressed) % CANDLE_STRUCT_SIZE)
    for off in range(0, usable, CANDLE_STRUCT_SIZE):
        t_sec, o_i, c_i, lo_i, hi_i, vol = struct.unpack(CANDLE_STRUCT_FMT, decompressed[off : off + CANDLE_STRUCT_SIZE])
        ts_s = (midnight + dt.timedelta(seconds=int(t_sec))).isoformat().replace("+00:00", "Z")
        rows.append((ts_s, price_fmt.format(o_i / price_div), price_fmt.format(hi_i / price_div), price_fmt.format(lo_i / price_div), price_fmt.format(c_i / price_div), f"{float(vol):.6f}".rstrip("0").rstrip(".")))
    return rows


def write_csv_gz(path: str, rows: List[Tuple[str, str, str, str, str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "open", "high", "low", "close", "volume"])
        writer.writerows(rows)


@dataclass(frozen=True)
class Job:
    instrument_id: str
    day: dt.date


def download_and_convert_one(job: Job, out_dir: str, side: str, overwrite: bool, timeout_s: int, retries: int, backoff_s: float, limiter: Optional[RateLimiter], user_agent: str, default_non_jpy_scale: int, jpy_scale: int) -> Tuple[Job, str]:
    instrument = job.instrument_id.lower()
    out_path = os.path.join(out_dir, instrument, f"{job.day.isoformat()}_{side.lower()}_m1.csv.gz")
    if not overwrite and os.path.exists(out_path):
        return job, "SKIP_EXISTS"
    url = build_candle_url(instrument, job.day, side)
    tick_scale = guess_tick_scale(instrument, default_non_jpy_scale, jpy_scale)
    last_err: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            if limiter:
                limiter.wait()
            response = _get_session(user_agent).get(url, timeout=timeout_s)
            if response.status_code == 404:
                return job, "NO_DATA_404"
            if response.status_code == 429:
                sleep_s = backoff_s * (2**attempt)
                time.sleep(sleep_s)
                last_err = f"429_RATE_LIMIT (slept {sleep_s:.1f}s)"
                continue
            response.raise_for_status()
            rows = decode_bi5_candles(response.content, job.day, tick_scale)
            if not rows:
                return job, "EMPTY_DECODED"
            write_csv_gz(out_path, rows)
            return job, f"OK ({len(rows)} rows)"
        except Exception as exc:  # noqa: BLE001 - report per-job failures and continue other jobs.
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(backoff_s * (2**attempt))
    return job, f"FAIL ({last_err})"


def resolve_instruments(universe: str, pairs_file: Optional[str], timeout_s: int, user_agent: str) -> List[str]:
    if universe == "majors":
        return list(DEFAULT_MAJORS)
    if universe == "file":
        if not pairs_file:
            raise ValueError("--pairs-file is required when --universe=file")
        return load_pairs_from_file(pairs_file)
    crosses = fetch_instrument_ids_from_dukascopy_node(DUKASCOPY_NODE_FX_CROSSES, timeout_s, user_agent)
    if universe == "crosses":
        return crosses
    if universe == "all":
        return sorted(set(DEFAULT_MAJORS).union(crosses))
    raise ValueError(f"Unknown universe: {universe!r}. Use majors|crosses|all|file")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD), inclusive")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD), exclusive")
    parser.add_argument("--out", default="data/dukascopy_fx_m1/raw", help="Output directory")
    parser.add_argument("--side", default="BID", choices=["BID", "ASK"], help="Price side to download")
    parser.add_argument("--universe", default="majors", choices=["majors", "crosses", "all", "file"])
    parser.add_argument("--pairs-file", default=None, help="File with one instrument id per line (for --universe=file)")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--rps", type=float, default=2.0, help="Global requests/sec limit")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout (seconds)")
    parser.add_argument("--retries", type=int, default=2, help="Retries per file on transient errors")
    parser.add_argument("--backoff", type=float, default=1.0, help="Base backoff seconds for retries")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--user-agent", default="dukascopy-m1-downloader/1.2", help="Custom User-Agent header")
    parser.add_argument("--tick-scale-non-jpy", type=int, default=5, help="Heuristic tick scale for non-JPY quotes")
    parser.add_argument("--tick-scale-jpy", type=int, default=3, help="Heuristic tick scale for JPY quote pairs")
    args = parser.parse_args(argv)

    start = parse_date(args.start)
    end = parse_date(args.end)
    if end <= start:
        print("ERROR: --end must be after --start (end is exclusive).", file=sys.stderr)
        return 2
    instruments = resolve_instruments(args.universe.lower(), args.pairs_file, args.timeout, args.user_agent)
    if not instruments:
        print("ERROR: instrument list resolved to empty.", file=sys.stderr)
        return 2
    days = list(daterange(start, end))
    jobs = [Job(inst, day) for inst in instruments for day in days]
    limiter = RateLimiter(args.rps) if args.rps > 0 else None
    print(f"Universe={args.universe} instruments={len(instruments)} days={len(days)} jobs={len(jobs)}")
    print(f"Output dir: {os.path.abspath(args.out)}")
    print(f"Side: {args.side} | workers={args.max_workers} | rps={args.rps}")
    ok = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(download_and_convert_one, job, args.out, args.side, args.overwrite, args.timeout, args.retries, args.backoff, limiter, args.user_agent, args.tick_scale_non_jpy, args.tick_scale_jpy) for job in jobs]
        for i, future in enumerate(as_completed(futures), start=1):
            job, status = future.result()
            ok += int(status.startswith("OK"))
            skipped += int(status.startswith(("SKIP", "NO_DATA", "EMPTY")))
            failed += int(not status.startswith(("OK", "SKIP", "NO_DATA", "EMPTY")))
            print(f"{job.instrument_id} {job.day}: {status}")
            if i % 50 == 0 or i == len(jobs):
                print(f"Progress: ok={ok} skipped={skipped} failed={failed} / {len(jobs)}")
    print(f"Done: ok={ok} skipped={skipped} failed={failed} total={len(jobs)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
