param(
    [string]$TargetDir = "external/audiocraft"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$TargetPath = Join-Path $RepoRoot $TargetDir
$PatchPath = Join-Path $RepoRoot "patches\audiocraft_music_dataset_structured.patch"
$RepoUrl = "https://github.com/facebookresearch/audiocraft.git"

if (-not (Test-Path $TargetPath)) {
    git clone --depth 1 $RepoUrl $TargetPath
}

try {
    git -C $TargetPath apply $PatchPath
} catch {
    Write-Host "Patch was not applied automatically. It may already be applied or require manual merge."
}

New-Item -ItemType Directory -Force -Path (Join-Path $TargetPath "config\conditioner") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $TargetPath "config\dset\audio") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $TargetPath "config\solver\musicgen") | Out-Null

Copy-Item -Force (Join-Path $RepoRoot "audiocraft_overrides\conditioner\*.yaml") (Join-Path $TargetPath "config\conditioner\")
Copy-Item -Force (Join-Path $RepoRoot "audiocraft_overrides\dset\audio\*.yaml") (Join-Path $TargetPath "config\dset\audio\")
Copy-Item -Force (Join-Path $RepoRoot "audiocraft_overrides\solver\musicgen\*.yaml") (Join-Path $TargetPath "config\solver\musicgen\")

Write-Host ""
Write-Host "AudioCraft is prepared in: $TargetPath"
Write-Host "Next steps:"
Write-Host "1. Open docs/audiocraft_patch.md"
Write-Host "2. Create a proper AUDIOCRAFT_DORA_DIR"
Write-Host "3. Install the cloned AudioCraft into your training environment"

