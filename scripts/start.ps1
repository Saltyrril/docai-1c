# Запуск сервиса docai для 1С: проверяет Ollama и поднимает HTTP-сервис на 127.0.0.1.
# Запуск из корня репозитория:  .\scripts\start.ps1  [-Port 8000] [-Config configs\extract.yaml]
param(
    [int]$Port = 8000,
    [string]$Config = "",
    [string]$OllamaUrl = "http://127.0.0.1:11434"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location (Split-Path -Parent $PSScriptRoot)

# 1. Ollama должна быть запущена: без неё работает только путь XML.
try {
    $version = Invoke-RestMethod -Uri "$OllamaUrl/api/version" -TimeoutSec 3
    Write-Host "Ollama работает, версия $($version.version)"
} catch {
    Write-Host "Ollama не отвечает на $OllamaUrl." -ForegroundColor Red
    Write-Host "Запустите её (приложение Ollama или команда 'ollama serve') и повторите." -ForegroundColor Red
    exit 1
}

# 2. Модель из настроек должна быть скачана (предупреждение, а не ошибка: XML модели не требует).
$model = "qwen3.5:2b-q4_K_M"
$configPath = if ($Config) { $Config } else { "configs\extract.yaml" }
if (Test-Path $configPath) {
    $line = Select-String -Path $configPath -Pattern '^model:\s*(\S+)' | Select-Object -First 1
    if ($line) { $model = $line.Matches[0].Groups[1].Value }
}
try {
    $tags = Invoke-RestMethod -Uri "$OllamaUrl/api/tags" -TimeoutSec 3
    if (-not ($tags.models | Where-Object { $_.name -eq $model -or $_.model -eq $model })) {
        Write-Host "Предупреждение: модель $model не скачана ('ollama pull $model')." -ForegroundColor Yellow
    }
} catch {
    Write-Host "Предупреждение: не удалось получить список моделей Ollama." -ForegroundColor Yellow
}

# 3. Порт не должен быть занят.
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    Write-Host "Порт $Port уже занят. Укажите другой: .\scripts\start.ps1 -Port 8001" -ForegroundColor Red
    exit 1
}

# 4. Сервис слушает только 127.0.0.1 (адрес в команде не настраивается).
Write-Host "Запуск: http://127.0.0.1:$Port/docs"
$serveArgs = @("run", "docai", "serve", "--port", $Port)
if ($Config) { $serveArgs += @("--config", $Config) }
& uv @serveArgs
exit $LASTEXITCODE
