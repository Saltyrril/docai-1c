# Проверка связи 1С → сервис без участия человека (P15).
#
# Собирает обработку и запускает 1С:Предприятие с ней. Обработка сама нажимает
# «Проверить связь» и «Проверить через файлы», пишет ответы в build\onec_smoke.txt
# и закрывает 1С. Сервис должен быть запущен (.\scripts\start.ps1).
#
#   .\scripts\onec_smoke.ps1                         # на служебной пустой базе build\onec_ib
#   .\scripts\onec_smoke.ps1 -Infobase "D:\Базы\Бух" # на вашей файловой базе
#
# Учебная версия разрешает один сеанс на базу: перед запуском с -Infobase закройте 1С
# с этой базой. Обработка ничего не записывает в базу, только читает ответ сервиса.
param(
    [string]$Platform = "",
    [string]$Infobase = "",
    [int]$TimeoutSec = 120
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
. (Join-Path $PSScriptRoot "onec_common.ps1")

$buildArgs = @{}
if ($Platform) { $buildArgs["Platform"] = $Platform }
& (Join-Path $PSScriptRoot "onec_build.ps1") @buildArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $Platform) { $Platform = Find-Platform }
if ($Infobase) { $Ib = (Resolve-Path $Infobase).Path } else { $Ib = Join-Path $Root "build\onec_ib" }
$Epf = Join-Path $Root "build\DocaiCheck.epf"
$Result = Join-Path $Root "build\onec_smoke.txt"
if (Test-Path $Result) { Remove-Item $Result }

Write-Host "База: $Ib"
Write-Host "Запускаю 1С:Предприятие с обработкой (окно закроется само)..."
$args1c = @("ENTERPRISE", "/F", "`"$Ib`"", "/Execute", "`"$Epf`"", "/C", "`"docai-smoke=$Result`"", "/DisableStartupDialogs", "/DisableStartupMessages")
$p = Start-Process -FilePath $Platform -ArgumentList $args1c -PassThru
if (-not $p.WaitForExit($TimeoutSec * 1000)) {
    Stop-Process -Id $p.Id -Force
    Write-Host "1С не закрылась за $TimeoutSec с." -ForegroundColor Red
}

if (-not (Test-Path $Result)) {
    Write-Host "Обработка не записала результат: $Result" -ForegroundColor Red
    # Ошибку компиляции или выполнения 1С пишет в журнал регистрации базы
    uv run python scripts\onec_errors.py $Ib --minutes 10
    exit 1
}
$text = Get-Content $Result -Encoding UTF8 -Raw
Write-Host $text
$okA = $text -match "== Способ А ==\r?\nСвязь есть"
$okB = $text -match "== Способ Б ==\r?\nСвязь есть"
Write-Host ("Способ А (HTTP): " + $(if ($okA) { "работает" } else { "НЕ работает" }))
Write-Host ("Способ Б (файлы): " + $(if ($okB) { "работает" } else { "НЕ работает" }))
if ($okA -or $okB) { exit 0 } else { exit 1 }
