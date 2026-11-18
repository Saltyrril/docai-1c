# Автопроверка обработки «Загрузка первичных документов (ИИ)» (P16) без участия человека.
#
# Собирает build\DocaiLoad.epf и запускает 1С:Предприятие с ней на указанной базе Бухгалтерии.
# Обработка сама отправляет документ в сервис, ждёт распознавания, открывает форму проверки,
# отправляет правки (/v1/jobs/{id}/feedback), пишет отчёт в build\onec_load_smoke.txt и
# закрывает 1С. Сервис должен быть запущен (.\scripts\start.ps1). В базу обработка не пишет.
#
#   .\scripts\onec_load_smoke.ps1 -Infobase "$env:USERPROFILE\Documents\1c\docai_bp_empty"
#   .\scripts\onec_load_smoke.ps1 -Infobase <папка базы> -User Admin -Document data\synth\pdf\00002_upd.pdf
#
# Учебная версия разрешает один сеанс на базу: перед запуском закройте 1С с этой базой.
param(
    [Parameter(Mandatory = $true)][string]$Infobase,
    [string]$Document = "data\synth\pdf\00001_bill.pdf",
    [string]$User = "",
    [string]$Platform = "",
    [int]$TimeoutSec = 300
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
. (Join-Path $PSScriptRoot "onec_common.ps1")

$buildArgs = @{ Name = @("DocaiLoad") }
if ($Platform) { $buildArgs["Platform"] = $Platform }
& (Join-Path $PSScriptRoot "onec_build.ps1") @buildArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $Platform) { $Platform = Find-Platform }
$Ib = (Resolve-Path $Infobase).Path
$Doc = (Resolve-Path $Document).Path
$Epf = Join-Path $Root "build\DocaiLoad.epf"
$Result = Join-Path $Root "build\onec_load_smoke.txt"
if (Test-Path $Result) { Remove-Item $Result }

Write-Host "База: $Ib"
Write-Host "Документ: $Doc"
Write-Host "Запускаю 1С:Предприятие с обработкой (окно закроется само, распознавание до минуты)..."
$args1c = @("ENTERPRISE", "/F", "`"$Ib`"")
if ($User) { $args1c += @("/N", "`"$User`"") }
$args1c += @("/Execute", "`"$Epf`"", "/C", "`"docai-load-smoke=$Result|$Doc`"", "/DisableStartupDialogs", "/DisableStartupMessages")
$proc = Start-Process -FilePath $Platform -ArgumentList $args1c -PassThru
if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
    Stop-Process -Id $proc.Id -Force
    Write-Host "1С не закрылась за $TimeoutSec с." -ForegroundColor Red
}

if (-not (Test-Path $Result)) {
    Write-Host "Обработка не записала результат: $Result" -ForegroundColor Red
    # Ошибки компиляции и выполнения 1С пишет в журнал регистрации базы
    uv run python scripts\onec_errors.py $Ib --minutes 15
    exit 1
}
$text = Get-Content $Result -Encoding UTF8 -Raw
Write-Host $text
$okUpload = $text -match "Состояние: done"
$okReview = $text -match "== Проверка =="
$okFeedback = $text -match "Правки: отправлено \d+"
Write-Host ("Отправка и распознавание: " + $(if ($okUpload) { "работает" } else { "НЕ работает" }))
Write-Host ("Форма проверки: " + $(if ($okReview) { "работает" } else { "НЕ работает" }))
Write-Host ("Правки в сервис: " + $(if ($okFeedback) { "работает" } else { "НЕ работает" }))
if ($okUpload -and $okReview -and $okFeedback) { exit 0 } else { exit 1 }
