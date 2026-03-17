from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from common import FIELD_ORDER, coerce_structured_metadata, read_json, write_json


GENRE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bhip[ -]?hop\b", "Hip-Hop"),
    (r"\bchillhop\b", "Chillhop"),
    (r"\blo[- ]?fi\b", "Lo-Fi"),
    (r"\br&b\b|\brnb\b", "R&B"),
    (r"\bsoul\b", "Soul"),
    (r"\bgospel\b", "Gospel"),
    (r"\bchoral\b|\bchoir\b", "Choral"),
    (r"\bballad\b", "Ballad"),
    (r"\bpop\b", "Pop"),
    (r"\brock\b", "Rock"),
    (r"\bheavy metal\b|\bmetal\b", "Metal"),
    (r"\balternative\b", "Alternative"),
    (r"\bindie\b", "Indie"),
    (r"\bfolk\b", "Folk"),
    (r"\bcountry\b", "Country"),
    (r"\bblues\b", "Blues"),
    (r"\bjazz\b", "Jazz"),
    (r"\bclassical\b", "Classical"),
    (r"\borchestral\b", "Orchestral"),
    (r"\bcinematic\b|\bsoundtrack\b|\bmovie\b|\bfilm\b", "Soundtrack"),
    (r"\btechno\b", "Techno"),
    (r"\bhouse\b", "House"),
    (r"\bedm\b|\bdance\b|\bclub\b", "Dance"),
    (r"\btrance\b", "Trance"),
    (r"\bambient\b", "Ambient"),
    (r"\belectronic\b|\bsynth\b", "Electronic"),
    (r"\bsynthwave\b", "Synthwave"),
    (r"\breggae\b", "Reggae"),
    (r"\blatin\b", "Latin"),
    (r"\bworld\b", "World"),
    (r"\bexperimental\b", "Experimental"),
    (r"\bdrum and bass\b|\bdnb\b", "Drum and Bass"),
)

MOOD_TERMS: tuple[str, ...] = (
    "epic",
    "heroic",
    "triumphant",
    "energetic",
    "uplifting",
    "euphoric",
    "relaxing",
    "nostalgic",
    "chill",
    "melancholic",
    "melancholy",
    "sad",
    "soulful",
    "romantic",
    "emotional",
    "mellow",
    "sentimental",
    "passionate",
    "groovy",
    "peaceful",
    "warm",
    "intimate",
    "dark",
    "gritty",
    "mysterious",
    "sinister",
    "rebellious",
    "aggressive",
    "devotional",
    "dramatic",
    "playful",
    "joyful",
)

INSTRUMENT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bchoir\b|\bchorus\b", "Choir"),
    (r"\bfemale vocal(?:ist)?s?\b|\bfemale voice\b|\bfemale singer\b", "Female vocal"),
    (r"\bmale vocal(?:ist)?s?\b|\bmale voice\b|\bmale singer\b", "Male vocal"),
    (r"\brap(?:ping)? vocals?\b|\brapper\b", "Rap vocals"),
    (r"\bnarrat(?:ion|or|ing)\b|\bspoken word\b", "Narration"),
    (r"\bacoustic guitar\b", "Acoustic guitar"),
    (r"\belectric guitar\b|\bdistortion guitars?\b|\bguitars?\b", "Guitar"),
    (r"\bpiano\b|\bkeys?\b|\bkeyboard\b", "Piano"),
    (r"\brhodes\b|\belectric piano\b", "Electric piano"),
    (r"\bstrings?\b|\bstring ensemble\b|\bviolin\b|\bcello\b", "Strings"),
    (r"\bbrass\b|\bhorns?\b|\btrombones?\b|\btrumpets?\b", "Brass"),
    (r"\bsub-bass\b|\bbass line\b|\bbass part\b|\bbass guitar\b|\bbass synth(?:esizer)?\b|\bbass\b(?!\s+category)", "Bass"),
    (r"\bdrums?\b|\bdrumming\b|\bkick\b|\bpercussion\b", "Drums"),
    (r"\bshaker\b|\bcymbal\b|\bclaps?\b|\btambourine\b|\btimpani\b", "Percussion"),
    (r"\bsaxophone\b|\bsax\b", "Saxophone"),
    (r"\bflute\b", "Flute"),
    (r"\bsynth(?:esizer)?\b|\banalog bass synthesizer\b", "Synthesizer"),
    (r"\barpeggio\b|\barpeggiated synth\b", "Arpeggiated synth"),
    (r"\borgan\b", "Organ"),
)

QUALITY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\blow quality\b|\bpoor audio quality\b|\bvery poor\b", "Low quality"),
    (r"\bnoisy\b|\bnoise\b", "Noisy"),
    (r"\blive performance\b|\blive recording\b|\blive congregation\b", "Live recording"),
    (r"\bstudio\b", "Studio recording"),
    (r"\blo[- ]?fi\b", "Lo-fi"),
    (r"\bdistorted\b|\boverdriven\b", "Distorted"),
    (r"\bclean mix\b|\bhigh fidelity\b|\bhi[- ]?fi\b", "High fidelity"),
    (r"\bunbalanced\b", "Unbalanced stereo image"),
    (r"\breverb\b|\breverberant\b", "Reverberant"),
    (r"\bwarm\b", "Warm production"),
    (r"\bmuffled\b", "Muffled high end"),
)

TEMPO_HINTS: tuple[str, ...] = (
    "slow tempo",
    "medium tempo",
    "mid tempo",
    "fast tempo",
    "upbeat",
    "laid-back",
    "steady rhythm",
    "waltz",
    "groove",
    "beat",
    "drumming rhythm",
    "four-on-the-floor",
    "swing",
    "march",
    "double kicks",
    "sixteenth",
)

PLACEHOLDER_TEXT = {"", "unspecified", "unknown", "n/a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill missing structured metadata JSONs from MusicCaps captions."
    )
    parser.add_argument("--audio-root", type=Path, default=Path("data/musiccaps/clips"))
    parser.add_argument("--overwrite-missing", action="store_true")
    parser.add_argument("--repair-placeholders", action="store_true")
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    text = normalize_space(text)
    if not text:
        return []
    return [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk.strip()]


def ordered_matches(text: str, patterns: tuple[tuple[str, str], ...]) -> list[str]:
    found: list[tuple[int, str]] = []
    lowered = text.lower()
    for pattern, label in patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            found.append((match.start(), label))
    found.sort(key=lambda item: item[0])
    unique: list[str] = []
    for _, label in found:
        if label not in unique:
            unique.append(label)
    return unique


def dedupe_instruments(labels: list[str]) -> list[str]:
    labels = list(labels)
    if "Acoustic guitar" in labels and "Guitar" in labels:
        labels.remove("Guitar")
    if "Electric piano" in labels and "Piano" in labels:
        labels.remove("Piano")
    if "Choir" in labels and "Female vocal" in labels:
        labels.remove("Female vocal")
    if "Choir" in labels and "Male vocal" in labels:
        labels.remove("Male vocal")
    return labels


def extract_description(caption: str) -> str:
    sentences = split_sentences(caption)
    if not sentences:
        return "Unspecified"
    if len(sentences) == 1:
        return sentences[0]
    first_two = " ".join(sentences[:2])
    return normalize_space(first_two)


def extract_mood(text: str) -> str:
    lowered = text.lower()
    found: list[str] = []
    for term in MOOD_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            label = term.capitalize() if term != "r&b" else "R&B"
            if label not in found:
                found.append(label)
    return ", ".join(found[:4]) if found else "Unspecified"


def extract_genres(text: str) -> list[str]:
    matches = ordered_matches(text, GENRE_PATTERNS)
    return matches[:5] if matches else ["Unspecified"]


def extract_vocal_presence(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\bno voices\b|\bno voice\b|\bno vocal\b|\bno vocal melody\b", lowered):
        return "None"
    if "instrumental" in lowered and not re.search(r"\bvocal|voice|singer|choir|rap\b", lowered):
        return "None"
    matches = dedupe_instruments(ordered_matches(text, INSTRUMENT_PATTERNS))
    vocal_labels = [label for label in matches if label in {"Choir", "Female vocal", "Male vocal", "Rap vocals", "Narration"}]
    if vocal_labels:
        if vocal_labels[0] == "Choir" and "children" in lowered:
            return "Children's choir"
        return vocal_labels[0]
    if re.search(r"\bvocal|voice|singer|choir\b", lowered):
        return "Vocals"
    return "None" if "instrumental" in lowered else "Unspecified"


def first_match_position(text: str, label: str) -> int | None:
    patterns = [pattern for pattern, current in INSTRUMENT_PATTERNS if current == label]
    lowered = text.lower()
    positions = [re.search(pattern, lowered, flags=re.IGNORECASE).start() for pattern in patterns if re.search(pattern, lowered, flags=re.IGNORECASE)]
    return min(positions) if positions else None


def extract_lead_instrument(text: str, vocal_presence: str) -> str:
    matches = dedupe_instruments(ordered_matches(text, INSTRUMENT_PATTERNS))
    if not matches:
        return "Unspecified"

    vocal_lead = vocal_presence if vocal_presence not in {"None", "Unspecified"} else None
    non_vocal = [label for label in matches if label not in {"Choir", "Female vocal", "Male vocal", "Rap vocals", "Narration"}]

    if vocal_lead and non_vocal:
        vocal_pos = first_match_position(text, vocal_lead if vocal_lead != "Children's choir" else "Choir")
        non_vocal_pos = first_match_position(text, non_vocal[0])
        if vocal_pos is not None and non_vocal_pos is not None and vocal_pos <= non_vocal_pos:
            return vocal_lead

    if non_vocal:
        return non_vocal[0]
    return vocal_lead or matches[0]


def extract_accompaniment(text: str, lead_instrument: str) -> str:
    matches = dedupe_instruments(ordered_matches(text, INSTRUMENT_PATTERNS))
    keep: list[str] = []
    lead_aliases = {lead_instrument}
    if lead_instrument == "Children's choir":
        lead_aliases.add("Choir")
    vocal_labels = {"Choir", "Female vocal", "Male vocal", "Rap vocals", "Narration"}
    for label in matches:
        if label in lead_aliases:
            continue
        if label in vocal_labels:
            continue
        if label not in keep:
            keep.append(label)
    if keep:
        return ", ".join(keep[:4])
    if "solo" in text.lower():
        return "None"
    return "Unspecified"


def sentence_with_keywords(text: str, keywords: tuple[str, ...]) -> str | None:
    best_sentence = None
    best_score = 0
    for sentence in split_sentences(text):
        probe = sentence.lower()
        score = sum(1 for keyword in keywords if keyword in probe)
        if score > best_score:
            best_score = score
            best_sentence = sentence
    return best_sentence if best_score > 0 else None


def extract_tempo(text: str) -> str:
    sentence = sentence_with_keywords(text, TEMPO_HINTS)
    if sentence:
        return sentence.rstrip(".")
    lowered = text.lower()
    if "slow" in lowered:
        return "Slow"
    if "medium tempo" in lowered or "mid tempo" in lowered:
        return "Medium tempo"
    if "fast" in lowered:
        return "Fast"
    return "Unspecified"


def extract_production_quality(text: str) -> str:
    matches = ordered_matches(text, QUALITY_PATTERNS)
    if matches:
        return ", ".join(matches[:4])
    sentence = sentence_with_keywords(
        text,
        ("quality", "recording", "recorded", "noisy", "distorted", "stereo", "reverb", "lo-fi", "muffled"),
    )
    if sentence:
        return sentence.rstrip(".")
    return "Unspecified"


def infer_metadata_from_caption(caption: str) -> dict[str, Any]:
    vocal_presence = extract_vocal_presence(caption)
    lead_instrument = extract_lead_instrument(caption, vocal_presence)
    metadata = {
        "description": extract_description(caption),
        "general_mood": extract_mood(caption),
        "genre_tags": extract_genres(caption),
        "lead_instrument": lead_instrument,
        "accompaniment": extract_accompaniment(caption, lead_instrument),
        "tempo_and_rhythm": extract_tempo(caption),
        "vocal_presence": vocal_presence,
        "production_quality": extract_production_quality(caption),
    }
    return coerce_structured_metadata(metadata)


def is_placeholder(field: str, value: Any) -> bool:
    if isinstance(value, list):
        if not value:
            return True
        return all(str(item).strip().lower() in PLACEHOLDER_TEXT for item in value)
    lowered = str(value).strip().lower()
    if field == "vocal_presence":
        return lowered in PLACEHOLDER_TEXT
    if field == "accompaniment":
        return lowered in PLACEHOLDER_TEXT
    return lowered in PLACEHOLDER_TEXT


def repair_metadata(existing: dict[str, Any], inferred: dict[str, Any]) -> tuple[dict[str, Any], int]:
    repaired = dict(existing)
    updates = 0
    for field in FIELD_ORDER:
        current_value = repaired.get(field)
        inferred_value = inferred[field]
        if is_placeholder(field, current_value) and not is_placeholder(field, inferred_value):
            repaired[field] = inferred_value
            updates += 1
    return coerce_structured_metadata(repaired), updates


def main() -> None:
    args = parse_args()
    audio_root = args.audio_root.resolve()
    wav_files = sorted(audio_root.glob("*.wav"))

    missing_written = 0
    repaired_files = 0
    repaired_fields = 0

    for wav_path in wav_files:
        source_path = wav_path.with_suffix(".source.json")
        target_path = wav_path.with_suffix(".json")
        if not source_path.exists():
            continue

        source = read_json(source_path)
        caption = normalize_space(str(source.get("caption", "")))
        if not caption:
            continue

        inferred = infer_metadata_from_caption(caption)

        if not target_path.exists() or args.overwrite_missing:
            write_json(target_path, inferred)
            missing_written += 1
            continue

        if args.repair_placeholders:
            existing = read_json(target_path)
            repaired, updates = repair_metadata(existing, inferred)
            if updates > 0:
                write_json(target_path, repaired)
                repaired_files += 1
                repaired_fields += updates

    print(
        {
            "audio_root": str(audio_root),
            "missing_written": missing_written,
            "repaired_files": repaired_files,
            "repaired_fields": repaired_fields,
        }
    )


if __name__ == "__main__":
    main()
