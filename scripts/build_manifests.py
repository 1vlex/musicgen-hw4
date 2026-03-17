from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import soundfile as sf
from tqdm import tqdm

from common import coerce_structured_metadata, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/valid AudioCraft manifests from downloaded MusicCaps clips."
    )
    parser.add_argument("--clips-root", type=Path, default=Path("data/musiccaps/clips"))
    parser.add_argument("--output-root", type=Path, default=Path("egs/musiccaps"))
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--respect-audioset-eval", action="store_true")
    parser.add_argument("--allow-missing-structured-json", action="store_true")
    return parser.parse_args()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    return lowered in {"1", "true", "yes", "y", "t"}


def choose_split(
    wav_path: Path,
    source_payload: dict[str, Any] | None,
    valid_ratio: float,
    respect_audioset_eval: bool,
) -> str:
    if respect_audioset_eval and source_payload and "is_audioset_eval" in source_payload:
        return "valid" if parse_bool(source_payload["is_audioset_eval"]) else "train"

    digest = hashlib.md5(wav_path.stem.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "valid" if bucket < valid_ratio else "train"


def audio_meta(wav_path: Path) -> dict[str, Any]:
    info = sf.info(str(wav_path))
    return {
        "path": str(wav_path.resolve()),
        "duration": float(info.frames) / float(info.samplerate),
        "sample_rate": int(info.samplerate),
    }


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    wav_files = sorted(args.clips_root.resolve().rglob("*.wav"))

    train_rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for wav_path in tqdm(wav_files, desc="Building manifests"):
        source_payload = None
        source_path = wav_path.with_suffix(".source.json")
        metadata_path = wav_path.with_suffix(".json")

        if source_path.exists():
            source_payload = read_json(source_path)

        if not args.allow_missing_structured_json:
            if not metadata_path.exists():
                skipped.append({"audio": str(wav_path), "reason": "missing structured json"})
                continue
            try:
                coerce_structured_metadata(read_json(metadata_path))
            except Exception as exc:  # noqa: BLE001
                skipped.append({"audio": str(wav_path), "reason": f"invalid structured json: {exc}"})
                continue

        try:
            row = audio_meta(wav_path)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"audio": str(wav_path), "reason": f"audio probe failed: {exc}"})
            continue

        split = choose_split(wav_path, source_payload, args.valid_ratio, args.respect_audioset_eval)
        if split == "valid":
            valid_rows.append(row)
        else:
            train_rows.append(row)

    output_root = args.output_root.resolve()
    write_jsonl_gz(output_root / "train" / "data.jsonl.gz", train_rows)
    write_jsonl_gz(output_root / "valid" / "data.jsonl.gz", valid_rows)

    summary = {
        "clips_root": str(args.clips_root.resolve()),
        "train_examples": len(train_rows),
        "valid_examples": len(valid_rows),
        "skipped_examples": len(skipped),
    }
    write_json(output_root / "summary.json", summary)
    if skipped:
        write_json(output_root / "skipped.json", skipped)

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
