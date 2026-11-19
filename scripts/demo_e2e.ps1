# Сквозная демонстрация (P17): Ollama -> сервис -> 10 документов через API -> сводка.
#
# Что делает:
#   1. Готовит data\demo\ - 10 документов разных видов и качества (копии из data\synth, data\datasets, build\p16_handoff).
#   2. Запускает Ollama, если она не работает, и ждёт ответа.
#   3. Запускает сервис docai, если он не работает, и проверяет GET /v1/health.
#   4. Прогоняет документы через API так же, как 1С (multipart, опрос задания), печатает сводку.
#   5. Останавливает только то, что запустил сам (если не указан -KeepRunning).
#
#   .\scripts\demo_e2e.ps1                          # полный прогон, 10 документов
#   .\scripts\demo_e2e.ps1 -KeepRunning             # оставить сервис работать для демонстрации в 1С
#   .\scripts\demo_e2e.ps1 -Only xml                # только документы из ЭДО (без модели, секунды)
#   .\scripts\demo_e2e.ps1 -Out reports\2026-11-19_p17_e2e\api_run.json
#
# Код возврата: 0 - все документы обработаны, 1 - есть ошибки, 2 - окружение не поднялось.
param(
    [int]$Port = 8000,
    [string]$DataDir = "",       # отдельная папка данных сервиса (если запускаем свой, а не используем работающий)
    [string]$OllamaUrl = "http://127.0.0.1:11434",
    [string]$Out = "",
    [ValidateSet("all", "xml")][string]$Only = "all",
    [switch]$KeepRunning,
    [int]$StartTimeoutSec = 60
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$startedOllama = $null
$startedService = $null

function Stop-Tree([System.Diagnostics.Process]$Process) {
    # uv запускает python дочерним процессом: останавливаем всё дерево.
    if ($Process -and -not $Process.HasExited) {
        & taskkill.exe /PID $Process.Id /T /F | Out-Null
    }
}

function Wait-Until([scriptblock]$Check, [int]$TimeoutSec) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try { if (& $Check) { return $true } } catch { }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Fail([string]$Message, [int]$Code = 2) {
    Write-Host $Message -ForegroundColor Red
    if (-not $KeepRunning) {
        Stop-Tree $script:startedService
        Stop-Tree $script:startedOllama
    }
    exit $Code
}

# --- 1. Демо-набор: 10 документов -------------------------------------------------------------
$demo = Join-Path $Root "data\demo"
New-Item -ItemType Directory -Force $demo | Out-Null
$set = @(
    # XML из ЭДО: путь без модели
    @("build\p16_handoff\testdata\demo_upd.xml", "01_upd_edo_zhi118.xml"),
    @("build\p16_handoff\testdata\demo_upd2.xml", "02_upd_edo_services.xml"),
    # PDF с текстовым слоем: путь text
    @("data\synth\pdf\00001_bill.pdf", "03_bill_pdf.pdf"),
    @("data\synth\pdf\00002_upd.pdf", "04_upd_pdf.pdf"),
    @("data\synth\pdf\00003_bill.pdf", "05_bill_pdf.pdf"),
    # Картинки разного качества: путь vision
    @("data\datasets\v1\images\00004_bill_p1_clean_200dpi.png", "06_bill_clean.png"),
    @("data\datasets\v1\images\00005_upd_p1_clean_200dpi.png", "07_upd_clean.png"),
    @("data\datasets\v1\images\00003_bill_p1_scan_200dpi.jpg", "08_bill_scan.jpg"),
    @("data\datasets\v1\images\00002_upd_p1_scan_200dpi.jpg", "09_upd_scan.jpg"),
    @("data\datasets\v1\images\00001_bill_p1_photo_200dpi.jpg", "10_bill_photo.jpg"),
    # Для показа в 1С (папка 1c\ в прогон через API не входит): УПД ЖИ-301, одна строка похожа на номенклатуру базы
    @("build\p17\xml\t1.xml", "1c\upd_edo_ZHI301.xml")
)
foreach ($pair in $set) {
    $src = Join-Path $Root $pair[0]
    $dst = Join-Path $demo $pair[1]
    New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
    if (Test-Path $dst) { continue }
    if (-not (Test-Path $src)) {
        Write-Host "Нет исходного файла $($pair[0]): пропускаю $($pair[1])." -ForegroundColor Yellow
        continue
    }
    Copy-Item $src $dst
}
if ($Only -eq "xml") {
    $runDir = Join-Path $Root "build\demo_xml"
    New-Item -ItemType Directory -Force $runDir | Out-Null
    Get-ChildItem $runDir | Remove-Item -Force
    Get-ChildItem $demo -Filter *.xml | Copy-Item -Destination $runDir
} else {
    $runDir = $demo
}
$count = @(Get-ChildItem $runDir -File).Count
if ($count -eq 0) { Fail "В $runDir нет документов." }
Write-Host "Демо-набор: $count документов в $runDir"

# --- 2. Ollama --------------------------------------------------------------------------------
$ollamaUp = { (Invoke-RestMethod -Uri "$OllamaUrl/api/version" -TimeoutSec 2).version }
try {
    Write-Host "Ollama уже работает, версия $(& $ollamaUp)"
} catch {
    Write-Host "Ollama не отвечает, запускаю 'ollama serve'..."
    $exe = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $exe) { Fail "Команда ollama не найдена. Установите Ollama или запустите её приложение." }
    $startedOllama = Start-Process -FilePath $exe.Source -ArgumentList "serve" -WindowStyle Hidden -PassThru
    if (-not (Wait-Until { & $ollamaUp } $StartTimeoutSec)) {
        Fail "Ollama не ответила за $StartTimeoutSec с."
    }
    Write-Host "Ollama запущена, версия $(& $ollamaUp)"
}

# --- 3. Сервис --------------------------------------------------------------------------------
$base = "http://127.0.0.1:$Port"
$healthUrl = "$base/v1/health"
$serviceUp = { (Invoke-RestMethod -Uri $healthUrl -TimeoutSec 3).status }
try {
    Write-Host "Сервис уже работает на порту $Port (статус $(& $serviceUp))"
} catch {
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        Fail "Порт $Port занят другой программой, не сервисом docai. Укажите другой: -Port 8001."
    }
    Write-Host "Запускаю сервис docai на порту $Port..."
    $serveArgs = @("run", "docai", "serve", "--port", $Port)
    if ($DataDir) { $serveArgs += @("--data-dir", $DataDir) }
    $log = Join-Path $Root "build\demo_serve.log"
    New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
    $startedService = Start-Process -FilePath "uv" -ArgumentList $serveArgs `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    if (-not (Wait-Until { & $serviceUp } $StartTimeoutSec)) {
        Fail "Сервис не ответил за $StartTimeoutSec с. Журнал: $log.err"
    }
}

$health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 5
Write-Host ("Здоровье: статус {0}, воркер {1}, модель {2} (скачана: {3}, в памяти: {4}), GPU свободно {5} МиБ" -f `
    $health.status, $health.worker_alive, $health.ollama.model, $health.ollama.model_available,
    $health.ollama.model_loaded, $health.gpu.vram_free_mib)
if (-not $health.worker_alive) { Fail "Воркер сервиса не работает." }
if ($health.status -ne "ok" -and $Only -ne "xml") {
    Fail "Сервис работает не полностью: $($health.ollama.error). Документы без текстового слоя не распознаются."
}

# --- 4. Прогон через API ----------------------------------------------------------------------
$pyArgs = @("run", "python", "scripts\demo_run.py", "--dir", $runDir, "--port", $Port)
if ($Out) { $pyArgs += @("--out", $Out) }
& uv @pyArgs
$code = $LASTEXITCODE

# --- 5. Уборка --------------------------------------------------------------------------------
if ($KeepRunning) {
    Write-Host "Сервис и Ollama оставлены работать (-KeepRunning). Swagger: $base/docs"
} else {
    if ($startedService) { Stop-Tree $startedService; Write-Host "Сервис, запущенный скриптом, остановлен." }
    if ($startedOllama) { Stop-Tree $startedOllama; Write-Host "Ollama, запущенная скриптом, остановлена." }
}
exit $code
