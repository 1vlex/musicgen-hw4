from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from common import FIELD_ORDER, coerce_structured_metadata, load_system_prompt, read_json, write_json


GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw MusicCaps captions into structured JSON sidecars using a Groq-compatible API."
    )
    parser.add_argument("--audio-root", type=Path, default=Path("data/musiccaps/clips"))
    parser.add_argument("--system-prompt-path", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--retry-attempts", type=int, default=6)
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    return parser.parse_args()


def build_user_prompt(caption: str) -> str:
    schema_lines = "\n".join(f'- "{field}"' for field in FIELD_ORDER)
    return (
        "Convert the following MusicCaps caption into structured metadata.\n\n"
        "Return a single JSON object with exactly these fields:\n"
        f"{schema_lines}\n\n"
        "Do not add any extra keys.\n\n"
        f"Caption:\n{caption.strip()}"
    )


def build_json_schema() -> dict[str, Any]:
    return {
        "name": "musiccaps_structured_metadata",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "description": {"type": "string"},
                "general_mood": {"type": "string"},
                "genre_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "lead_instrument": {"type": "string"},
                "accompaniment": {"type": "string"},
                "tempo_and_rhythm": {"type": "string"},
                "vocal_presence": {"type": "string"},
                "production_quality": {"type": "string"},
            },
            "required": list(FIELD_ORDER),
        },
    }


def resolve_provider_credentials() -> tuple[str, str]:
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_api_key:
        return "groq", groq_api_key

    xai_api_key = os.getenv("XAI_API_KEY", "").strip()
    if xai_api_key.startswith("gsk_"):
        # Support the user's current console workflow where a Groq key was exported
        # into XAI_API_KEY by mistake.
        return "groq", xai_api_key

    if xai_api_key:
        raise RuntimeError(
            "XAI_API_KEY is set, but this script is currently configured for Groq-compatible keys. "
            "If your key starts with 'gsk_', export it as GROQ_API_KEY or keep it in XAI_API_KEY."
        )

    raise RuntimeError(
        "No compatible API key found. Set GROQ_API_KEY in PowerShell, or set XAI_API_KEY to a Groq-style key starting with 'gsk_'."
    )


def caption_to_metadata(
    api_key: str,
    model: str,
    system_prompt: str,
    caption: str,
    temperature: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(caption)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": build_json_schema(),
        },
    }
    response = requests.post(
        GROQ_ENDPOINT,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout_seconds,
    )

    if response.status_code >= 400:
        raise RuntimeError(f"Groq API error {response.status_code}: {response.text}")

    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Unexpected Groq response: {json.dumps(data, ensure_ascii=True)[:1000]}") from exc

    if not content:
        raise RuntimeError(f"Empty model response: {json.dumps(data, ensure_ascii=True)[:1000]}")

    return coerce_structured_metadata(json.loads(content))


def parse_retry_after_seconds(error_text: str) -> float | None:
    match = re.search(r"Please try again in ([0-9.]+)(ms|s)", error_text)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "ms":
        return max(0.5, value / 1000.0)
    return max(0.5, value)


def iter_audio_files(audio_root: Path) -> list[Path]:
    return sorted(audio_root.rglob("*.wav"))


def main() -> None:
    load_dotenv()
    args = parse_args()

    _, api_key = resolve_provider_credentials()
    model = args.model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    system_prompt = load_system_prompt(str(args.system_prompt_path) if args.system_prompt_path else None)

    audio_files = iter_audio_files(args.audio_root.resolve())
    if args.max_items is not None:
        audio_files = audio_files[: args.max_items]

    failures: list[dict[str, Any]] = []
    written = 0

    for wav_path in tqdm(audio_files, desc="Enriching metadata"):
        source_path = wav_path.with_suffix(".source.json")
        target_path = wav_path.with_suffix(".json")

        if target_path.exists() and not args.overwrite:
            continue
        if not source_path.exists():
            failures.append({"audio": str(wav_path), "reason": "missing .source.json"})
            continue

        source_payload = read_json(source_path)
        caption = str(source_payload.get("caption", "")).strip()
        if not caption:
            failures.append({"audio": str(wav_path), "reason": "empty caption"})
            continue

        try:
            last_error: Exception | None = None
            metadata: dict[str, Any] | None = None
            for attempt in range(1, args.retry_attempts + 1):
                try:
                    metadata = caption_to_metadata(
                        api_key=api_key,
                        model=model,
                        system_prompt=system_prompt,
                        caption=caption,
                        temperature=args.temperature,
                        timeout_seconds=args.timeout_seconds,
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    error_text = str(exc)
                    if "Groq API error 429" not in error_text or attempt == args.retry_attempts:
                        raise
                    retry_after = parse_retry_after_seconds(error_text) or args.retry_sleep_seconds
                    time.sleep(retry_after)

            if metadata is None:
                raise RuntimeError(str(last_error) if last_error else "metadata generation failed")
            write_json(target_path, metadata)
            written += 1
        except Exception as exc:  # noqa: BLE001
            failures.append({"audio": str(wav_path), "reason": str(exc)})

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    if failures:
        failure_log = args.audio_root.resolve() / "metadata_failures.json"
        write_json(failure_log, failures)
        print(f"Finished with {len(failures)} failures. See: {failure_log}")
    else:
        print("Finished without failures.")

    print(f"Structured metadata written: {written}")


if __name__ == "__main__":
    main()
