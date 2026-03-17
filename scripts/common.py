from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FIELD_ORDER = (
    "description",
    "general_mood",
    "genre_tags",
    "lead_instrument",
    "accompaniment",
    "tempo_and_rhythm",
    "vocal_presence",
    "production_quality",
)

LIST_FIELDS = {"genre_tags"}

SCHEMA_EXAMPLE = {
    "description": "string",
    "general_mood": "string",
    "genre_tags": ["string"],
    "lead_instrument": "string",
    "accompaniment": "string",
    "tempo_and_rhythm": "string",
    "vocal_presence": "string",
    "production_quality": "string",
}

DEFAULT_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "metadata_system_prompt.txt"
)


def load_system_prompt(path: str | None = None) -> str:
    prompt_path = Path(path) if path else DEFAULT_SYSTEM_PROMPT_PATH
    return prompt_path.read_text(encoding="utf-8").strip()


def _normalize_text(value: Any, fallback: str = "Unspecified") -> str:
    text = "" if value is None else str(value).strip()
    return text or fallback


def coerce_structured_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in FIELD_ORDER if field not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    cleaned: dict[str, Any] = {}
    for field in FIELD_ORDER:
        value = payload[field]
        if field in LIST_FIELDS:
            if isinstance(value, str):
                items = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
            elif isinstance(value, list):
                items = [str(chunk).strip() for chunk in value if str(chunk).strip()]
            else:
                raise ValueError(f"Field '{field}' must be a list of strings, got: {type(value)}")
            if not items:
                items = ["Unspecified"]
            cleaned[field] = items
        else:
            fallback = "None" if field == "vocal_presence" else "Unspecified"
            cleaned[field] = _normalize_text(value, fallback=fallback)

    return cleaned


def metadata_to_condition_map(metadata: dict[str, Any]) -> dict[str, str]:
    cooked = coerce_structured_metadata(metadata)
    conditions: dict[str, str] = {}
    for field in FIELD_ORDER:
        value = cooked[field]
        if isinstance(value, list):
            conditions[field] = ", ".join(value)
        else:
            conditions[field] = value
    return conditions


def metadata_to_single_prompt(metadata: dict[str, Any]) -> str:
    conditions = metadata_to_condition_map(metadata)
    ordered_lines = [
        conditions["description"],
        f"mood: {conditions['general_mood']}",
        f"genre tags: {conditions['genre_tags']}",
        f"lead instrument: {conditions['lead_instrument']}",
        f"accompaniment: {conditions['accompaniment']}",
        f"tempo and rhythm: {conditions['tempo_and_rhythm']}",
        f"vocal presence: {conditions['vocal_presence']}",
        f"production quality: {conditions['production_quality']}",
    ]
    return ". ".join(part.rstrip(".") for part in ordered_lines if part).strip()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

