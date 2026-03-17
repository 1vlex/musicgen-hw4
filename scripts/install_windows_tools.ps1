$ErrorActionPreference = "Stop"

winget install --id Gyan.FFmpeg.Essentials -e --accept-package-agreements --accept-source-agreements
winget install --id yt-dlp.yt-dlp -e --accept-package-agreements --accept-source-agreements

Write-Host ""
Write-Host "Открой новый PowerShell и проверь:"
Write-Host "ffmpeg -version"
Write-Host "yt-dlp --version"
