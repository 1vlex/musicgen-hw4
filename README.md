# Fine-tuning MusicGen on MusicCaps

Репозиторий содержит пайплайн для fine-tuning `MusicGen` на `MusicCaps` с использованием структурированных метаданных.

В проекте есть:

- скачивание 10-секундных аудиофрагментов из MusicCaps
- преобразование caption в structured JSON через LLM
- сборка `train` и `valid` manifests для AudioCraft
- патч для AudioCraft
- запуск fine-tuning `musicgen-small`
- инференс по тестовым structured prompts

## Структура репозитория

- [scripts](./scripts)
- [prompts](./prompts)
- [patches](./patches)
- [audiocraft_overrides](./audiocraft_overrides)
- [report.md](./report.md)

## Установка

### 1. Установить зависимости проекта

```powershell
cd "E:\Fine-tuning MusicGen"
py -3.10 -m pip install --upgrade pip setuptools wheel
py -3.10 -m pip install -r requirements.txt
```

### 2. Установить `ffmpeg` и `yt-dlp`

```powershell
winget install --id Gyan.FFmpeg.Essentials -e --accept-package-agreements --accept-source-agreements
winget install --id yt-dlp.yt-dlp -e --accept-package-agreements --accept-source-agreements
```

### 3. Клонировать и установить AudioCraft

```powershell
cd "E:\Fine-tuning MusicGen"
git clone https://github.com/facebookresearch/audiocraft external/audiocraft
cd external/audiocraft
git apply "..\..\patches\audiocraft_music_dataset_structured.patch"
py -3.10 -m pip install -e .
cd "..\.."
```

## Переменные окружения

Шаблон лежит в [.env.example](./.env.example).

Перед запуском команд в PowerShell нужно экспортировать:

```powershell
$env:GROQ_API_KEY = "YOUR_GROQ_API_KEY"
$env:GROQ_MODEL = "openai/gpt-oss-20b"
$env:COMET_API_KEY = "YOUR_COMET_API_KEY"
$env:COMET_WORKSPACE = "YOUR_COMET_WORKSPACE"
$env:COMET_PROJECT_NAME = "musicgen-hw4"
$env:USER = $env:USERNAME
$env:MUSICGEN_PROJECT_ROOT = "E:\Fine-tuning MusicGen"
$env:AUDIOCRAFT_DORA_DIR = "E:\Fine-tuning MusicGen\outputs\dora"
$env:AUDIOCRAFT_TEAM = "default"
$env:AUDIOCRAFT_CLUSTER = "default"
```

## Подготовка датасета

### 1. Скачать аудиофрагменты

```powershell
cd "E:\Fine-tuning MusicGen"
py -3.10 scripts/download_musiccaps.py --output-root data/musiccaps/clips --max-items 258
```

Результат:

- `*.wav`
- `*.source.json`

### 2. Построить structured metadata

```powershell
cd "E:\Fine-tuning MusicGen"
py -3.10 scripts/enrich_metadata.py --audio-root data/musiccaps/clips
```

Для каждого клипа сохраняется `.json` следующей схемы:

```json
{
  "description": "string",
  "general_mood": "string",
  "genre_tags": ["string"],
  "lead_instrument": "string",
  "accompaniment": "string",
  "tempo_and_rhythm": "string",
  "vocal_presence": "string",
  "production_quality": "string"
}
```

### 3. Собрать manifests для AudioCraft

```powershell
cd "E:\Fine-tuning MusicGen"
py -3.10 scripts/build_manifests.py --clips-root data/musiccaps/clips --output-root egs/musiccaps
```

## Запуск обучения

```powershell
cd "E:\Fine-tuning MusicGen\external\audiocraft"
py -3.10 -m dora -P audiocraft run solver=musicgen/musicgen_small_musiccaps_structured_16gb --clear
```

Используемый конфиг:

- [musicgen_small_musiccaps_structured_16gb.yaml](./audiocraft_overrides/solver/musicgen/musicgen_small_musiccaps_structured_16gb.yaml)

## Запуск инференса

Ссылка на архив с лучшей моделью и итоговыми аудио:

- [Тык](https://drive.google.com/file/d/1XY-N3BaRl5-VglwYOghMYD57SG7zvi81/view?usp=sharing)

После распаковки архива должна получиться папка с файлами:

- `state_dict.bin`
- `compression_state_dict.bin`

Команда генерации:

```powershell
cd "E:\Fine-tuning MusicGen"
py -3.10 scripts/run_structured_inference.py --model-id "E:\path\to\musicgen_small_structured_best" --prompts-file prompts/test_prompts.json --output-dir outputs/test_prompts_best --duration 10
```

Результат:

- `prompt_1.wav`
- `prompt_2.wav`
- `prompt_3.wav`
- `prompt_4.wav`
- `prompt_5.wav`

Подробности по логам обучения, трудностям и итогам приведены в [Тык](./report.md).
