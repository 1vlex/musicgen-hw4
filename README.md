# Fine-tuning MusicGen on MusicCaps

Этот репозиторий содержит пайплайн для домашнего задания по fine-tuning `MusicGen` из `AudioCraft` на датасете `MusicCaps`.

В проекте реализованы:

- скачивание 10-секундных аудиофрагментов из MusicCaps без загрузки полного видео
- обогащение исходных caption до структурированного JSON через LLM
- сборка `train` и `valid` manifests для AudioCraft
- модификация пайплайна AudioCraft для использования новых полей
- fine-tuning `musicgen-small`
- инференс по тестовым структурированным промптам

## Содержимое репозитория

- [scripts](./scripts) - подготовка данных, обогащение метаданных, инференс
- [prompts](./prompts) - системный промпт для LLM и тестовые промпты для генерации
- [patches/audiocraft_music_dataset_structured.patch](./patches/audiocraft_music_dataset_structured.patch) - патч для локального клона AudioCraft
- [audiocraft_overrides](./audiocraft_overrides) - Hydra overrides для обучения
- [report.md](./report.md) - краткий отчет по эксперименту

Локальный клон `AudioCraft` не включен в репозиторий, чтобы не раздувать размер проекта. Вместо этого приложен патч и конфиги.

## Что понадобится

- Python 3.10
- PyTorch с CUDA под вашу видеокарту
- `ffmpeg`
- `yt-dlp`
- Groq API key для LLM-обогащения метаданных
- Comet ML API key для логирования обучения

## Установка

### 1. Установить Python-зависимости проекта

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

Пример лежит в [.env.example](./.env.example).

Используемые переменные:

```env
GROQ_API_KEY=...
GROQ_MODEL=openai/gpt-oss-20b
COMET_API_KEY=...
COMET_WORKSPACE=YOUR_COMET_WORKSPACE
COMET_PROJECT_NAME=musicgen-hw4
```

## Подготовка датасета

### 1. Скачать аудиофрагменты MusicCaps

Скрипт использует `yt-dlp` для получения прямой ссылки на аудиопоток и `ffmpeg` для скачивания ровно 10 секунд в `.wav`.

```powershell
cd "E:\Fine-tuning MusicGen"
py -3.10 scripts/download_musiccaps.py --output-root data/musiccaps/clips --max-items 258
```

Результат:

- `*.wav`
- `*.source.json`

### 2. Обогатить caption до структурированного JSON

Скрипт вызывает Groq-compatible API и сохраняет sidecar JSON рядом с каждым `.wav`.

```powershell
cd "E:\Fine-tuning MusicGen"
py -3.10 scripts/enrich_metadata.py --audio-root data/musiccaps/clips
```

Схема для каждого трека:

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

В моем запуске получилось:

- `train`: 248 клипов
- `valid`: 10 клипов

## Что изменено в AudioCraft

Основная идея была такой:

- добавить новые поля в `MusicInfo`
- читать structured JSON при загрузке датасета
- передавать эти поля в текстовый энкодер

В текущей реализации structured поля используются через сериализацию в единый текстовый prompt. Это сделано для совместимости с `musicgen-small` и ограничения по памяти.

То есть модель получает не только исходный `description`, но и дополнительные поля:

- `general_mood`
- `genre_tags`
- `lead_instrument`
- `accompaniment`
- `tempo_and_rhythm`
- `vocal_presence`
- `production_quality`

## Запуск обучения

Обучение запускалось на `musicgen-small`.

Ключевые параметры:

- batch size: 4
- learning rate: `5e-5`
- epochs: 10
- early stopping patience: 4
- early stopping min epochs: 4
- updates per epoch: 62
- segment duration: 10 секунд
- warmup: 50 steps
- `merge_text_p = 0.85`
- `drop_desc_p = 0.20`
- `drop_other_p = 0.80`

Команда запуска:

```powershell
cd "E:\Fine-tuning MusicGen\external\audiocraft"

$env:USER = $env:USERNAME
$env:MUSICGEN_PROJECT_ROOT = "E:\Fine-tuning MusicGen"
$env:AUDIOCRAFT_DORA_DIR = "E:\Fine-tuning MusicGen\outputs\dora"
$env:AUDIOCRAFT_TEAM = "default"
$env:AUDIOCRAFT_CLUSTER = "default"
$env:COMET_API_KEY = "YOUR_COMET_API_KEY"
$env:COMET_WORKSPACE = "YOUR_COMET_WORKSPACE"
$env:COMET_PROJECT_NAME = "musicgen-hw4"

py -3.10 -m dora -P audiocraft run solver=musicgen/musicgen_small_musiccaps_structured_16gb --clear
```

## Запуск инференса

Архив с лучшей моделью и итоговыми примерами доступен по ссылке:

- Google Drive: <https://drive.google.com/file/d/1XY-N3BaRl5-VglwYOghMYD57SG7zvi81/view?usp=sharing>

Для воспроизведения инференса нужно распаковать веса так, чтобы получилась папка с файлами:

- `state_dict.bin`
- `compression_state_dict.bin`

После этого можно запустить генерацию по тестовым промптам:

```powershell
cd "E:\Fine-tuning MusicGen"
py -3.10 scripts/run_structured_inference.py --model-id "E:\path\to\musicgen_small_structured_best" --prompts-file prompts/test_prompts.json --output-dir outputs/test_prompts_best --duration 10
```

Скрипт сохранит:

- `prompt_1.wav`
- `prompt_2.wav`
- `prompt_3.wav`
- `prompt_4.wav`
- `prompt_5.wav`

## Логи обучения

Обучение отражено в двух Comet ML экспериментах:

- первая часть: <https://www.comet.com/1vlex/musicgen-hw4/c75ca34d9dcb44a880ab872a052a74f4?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&viewId=new&xAxis=step>
- продолжение: <https://www.comet.com/1vlex/musicgen-hw4/799739ecccc348828d3459f26804574a?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&viewId=new&xAxis=step>

## Что приложено к сдаче

- репозиторий с кодом подготовки данных и инференса
- патч для AudioCraft
- ссылка на веса модели
- ссылка на итоговые аудиофайлы
- отчет в [report.md](./report.md)
