from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

from common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download 10-second MusicCaps clips with yt-dlp + ffmpeg without full video download."
    )
    parser.add_argument("--dataset-name", default="google/MusicCaps")
    parser.add_argument("--split", default="train")
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("data/musiccaps/clips"))
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--clip-duration", type=float, default=10.0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--yt-dlp-bin", default="yt-dlp")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--cookies", type=Path, default=None)
    parser.add_argument("--command-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--yt-dlp-retries", type=int, default=3)
    parser.add_argument("--ffmpeg-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required binary '{name}' was not found in PATH. Install it first and retry."
        )


def pick(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def sample_stem(row: dict[str, Any], index: int) -> str:
    musiccaps_id = pick(row, "musiccaps_id", "id", default=index)
    ytid = pick(row, "ytid", default="unknown")
    return f"{int(musiccaps_id):06d}_{ytid}"


def run_command(command: list[str], timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        joined = " ".join(command[:3])
        raise RuntimeError(f"Command timed out after {timeout_seconds:.1f}s: {joined}") from exc


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.metadata_csv:
        with args.metadata_csv.resolve().open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    dataset = load_dataset(args.dataset_name, split=args.split)
    return [dict(dataset[index]) for index in range(len(dataset))]


def resolve_stream_url(args: argparse.Namespace, youtube_url: str) -> str:
    command = [
        args.yt_dlp_bin,
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "--extractor-retries",
        str(args.yt_dlp_retries),
        "--socket-timeout",
        "15",
        "-f",
        "ba/bestaudio",
        "-g",
        youtube_url,
    ]
    if args.cookies:
        command.extend(["--cookies", str(args.cookies)])

    last_error = "yt-dlp failed"
    for attempt in range(1, args.yt_dlp_retries + 1):
        result = run_command(command, timeout_seconds=args.command_timeout_seconds)
        if result.returncode == 0:
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if lines:
                return lines[0]
            last_error = "yt-dlp returned an empty stream URL"
        else:
            last_error = result.stderr.strip() or result.stdout.strip() or "yt-dlp failed"

        if attempt < args.yt_dlp_retries:
            time.sleep(args.retry_sleep_seconds)

    raise RuntimeError(f"yt-dlp failed after {args.yt_dlp_retries} attempts: {last_error}")


def download_clip(
    args: argparse.Namespace,
    stream_url: str,
    output_wav: Path,
    start_seconds: float,
) -> None:
    temp_wav = output_wav.with_name(f"{output_wav.name}.part")
    command = [
        args.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start_seconds),
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_delay_max",
        "10",
        "-i",
        stream_url,
        "-t",
        str(args.clip_duration),
        "-ar",
        str(args.sample_rate),
        "-ac",
        str(args.channels),
        "-vn",
        "-sn",
        "-dn",
        "-f",
        "wav",
        "-y",
        str(temp_wav),
    ]

    last_error = "ffmpeg failed"
    for attempt in range(1, args.ffmpeg_retries + 1):
        result = run_command(command, timeout_seconds=args.command_timeout_seconds)
        if result.returncode == 0 and temp_wav.exists():
            temp_wav.replace(output_wav)
            return

        last_error = result.stderr.strip() or result.stdout.strip() or "ffmpeg failed"
        if temp_wav.exists():
            temp_wav.unlink()
        if attempt < args.ffmpeg_retries:
            time.sleep(args.retry_sleep_seconds)

    raise RuntimeError(f"ffmpeg failed after {args.ffmpeg_retries} attempts: {last_error}")


def process_item(
    args: argparse.Namespace,
    output_root: Path,
    index: int,
    row: dict[str, Any],
) -> dict[str, Any]:
    stem = sample_stem(row, index)
    wav_path = output_root / f"{stem}.wav"
    source_path = output_root / f"{stem}.source.json"

    if wav_path.exists() and source_path.exists() and not args.overwrite:
        return {"status": "skipped"}

    ytid = pick(row, "ytid")
    caption = pick(row, "caption", default="")
    if not ytid:
        return {
            "status": "failed",
            "failure": {"index": index, "reason": "missing ytid"},
        }

    start_seconds = float(pick(row, "start_s", "start_seconds", default=0.0))
    youtube_url = f"https://www.youtube.com/watch?v={ytid}"

    try:
        stream_url = resolve_stream_url(args, youtube_url)
        download_clip(args, stream_url, wav_path, start_seconds)
        payload = {
            **row,
            "youtube_url": youtube_url,
            "requested_start_s": start_seconds,
            "requested_duration_s": args.clip_duration,
            "caption": caption,
        }
        write_json(source_path, payload)
        return {"status": "downloaded"}
    except Exception as exc:  # noqa: BLE001
        if wav_path.exists():
            wav_path.unlink()
        return {
            "status": "failed",
            "failure": {
                "index": index,
                "ytid": ytid,
                "caption": caption,
                "reason": str(exc),
            },
        }


def main() -> None:
    args = parse_args()
    ensure_binary(args.yt_dlp_bin)
    ensure_binary(args.ffmpeg_bin)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args)
    start = args.start_index
    stop = len(rows) if args.max_items is None else min(len(rows), start + args.max_items)

    failures: list[dict[str, Any]] = []
    downloaded = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(process_item, args, output_root, index, rows[index]): index
            for index in range(start, stop)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading MusicCaps"):
            result = future.result()
            status = result["status"]
            if status == "downloaded":
                downloaded += 1
            elif status == "skipped":
                skipped += 1
            else:
                failures.append(result["failure"])

    if failures:
        failure_log = output_root / "download_failures.json"
        write_json(failure_log, failures)
        print(f"Completed with {len(failures)} failures. See: {failure_log}")
    else:
        print(f"Completed successfully. Audio clips are in: {output_root}")

    print(
        f"Summary: downloaded={downloaded}, skipped={skipped}, failed={len(failures)}, "
        f"processed={stop - start}"
    )


if __name__ == "__main__":
    main()
