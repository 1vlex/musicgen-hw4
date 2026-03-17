from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from common import coerce_structured_metadata, metadata_to_condition_map, metadata_to_single_prompt, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate audio for structured prompts using a fine-tuned MusicGen/AudioCraft model."
    )
    parser.add_argument("--model-id", required=True, help="HF repo id or local exported model directory.")
    parser.add_argument("--prompts-file", type=Path, default=Path("prompts/test_prompts.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/test_prompts"))
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-coef", type=float, default=3.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force-concatenate", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def load_prompts(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Prompts file must contain a JSON list.")
    return payload


def save_waveform(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().cpu().float()
    if wav.dim() == 2 and wav.shape[0] == 1:
        sf.write(str(path), wav.squeeze(0).numpy(), sample_rate)
        return
    if wav.dim() == 2:
        sf.write(str(path), wav.transpose(0, 1).numpy(), sample_rate)
        return
    raise ValueError(f"Unexpected waveform shape: {tuple(wav.shape)}")


def build_empty_wav_condition(device: str, sample_rate: int):
    from audiocraft.modules.conditioners import WavCondition

    return WavCondition(
        wav=torch.zeros((1, 1, 1), device=device),
        length=torch.tensor([0], device=device),
        sample_rate=[sample_rate],
        path=[None],
        seek_time=[0.0],
    )


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from audiocraft.models import MusicGen
        from audiocraft.modules.conditioners import ConditioningAttributes
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "AudioCraft is not installed in the current environment. Install your patched clone first."
        ) from exc

    prompts = load_prompts(args.prompts_file.resolve())
    model = MusicGen.get_pretrained(args.model_id, device=device)
    model.set_generation_params(
        duration=args.duration,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        cfg_coef=args.cfg_coef,
        use_sampling=True,
    )

    text_conditions = list(model.lm.condition_provider.text_conditions)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for item in prompts:
        metadata = coerce_structured_metadata(item["metadata"])
        filename = item.get("filename") or f"prompt_{item.get('id', 'x')}.wav"
        wav_path = output_dir / filename
        json_path = wav_path.with_suffix(".json")

        if args.force_concatenate or text_conditions == ["description"]:
            merged_prompt = metadata_to_single_prompt(metadata)
            generated = model.generate([merged_prompt], progress=args.progress)
            wav = generated[0]
        else:
            attr = ConditioningAttributes(text={})
            condition_map = metadata_to_condition_map(metadata)
            for key in text_conditions:
                attr.text[key] = condition_map.get(key)

            if "self_wav" in model.lm.condition_provider.conditioners:
                attr.wav["self_wav"] = build_empty_wav_condition(device=device, sample_rate=model.sample_rate)

            tokens = model._generate_tokens([attr], prompt_tokens=None, progress=args.progress)
            wav = model.generate_audio(tokens)[0]

        save_waveform(wav_path, wav, model.sample_rate)
        write_json(json_path, metadata)
        print(f"Saved {wav_path}")


if __name__ == "__main__":
    main()

