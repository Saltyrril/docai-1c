# Сборка внешней обработки 1С из исходников onec\src в build\DocaiCheck.epf (P15, D54).
#
#   .\scripts\onec_build.ps1            # собрать .epf из onec\src
#   .\scripts\onec_build.ps1 -Dump      # обратно: выгрузить build\DocaiCheck.epf в onec\src
#                                       # (после правок обработки в Конфигураторе)
#
# Конфигуратор в пакетном режиме требует информационную базу. Скрипт создаёт пустую
# служебную базу build\onec_ib и работает только с ней: ваши базы не открываются.
param(
    [switch]$Dump,
    [string]$Platform = "",
    [string]$Epf = "build\DocaiCheck.epf"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
. (Join-Path $PSScriptRoot "onec_common.ps1")

function Invoke-Designer([string[]]$Arguments, [string]$Log) {
    if (Test-Path $Log) { Remove-Item $Log }
    $all = @("DESIGNER") + $Arguments + @("/DisableStartupDialogs", "/DisableStartupMessages", "/Out", "`"$Log`"")
    $p = Start-Process -FilePath $Platform -ArgumentList $all -Wait -PassThru -WindowStyle Hidden
    if (Test-Path $Log) { Get-Content $Log -Encoding Default | Write-Host }
    return $p.ExitCode
}

if (-not $Platform) { $Platform = Find-Platform }
if (-not $Platform -or -not (Test-Path $Platform)) {
    Write-Host "Не найдена платформа 1С (1cv8t.exe или 1cv8.exe в Program Files). Укажите -Platform <путь>." -ForegroundColor Red
    exit 1
}
Write-Host "Платформа: $Platform"

New-Item -ItemType Directory -Force "build" | Out-Null
$Ib = Join-Path $Root "build\onec_ib"
$Src = Join-Path $Root "onec\src\DocaiCheck.xml"
$EpfPath = Join-Path $Root $Epf

# Служебная пустая база: нужна Конфигуратору, данные в ней не хранятся.
if (-not (Test-Path (Join-Path $Ib "1Cv8.1CD"))) {
    Write-Host "Создаю служебную базу $Ib"
    $log = Join-Path $Root "build\onec_create.log"
    if (Test-Path $log) { Remove-Item $log }
    $p = Start-Process -FilePath $Platform -ArgumentList @("CREATEINFOBASE", "File=`"$Ib`"", "/DisableStartupDialogs", "/Out", "`"$log`"") -Wait -PassThru -WindowStyle Hidden
    if (Test-Path $log) { Get-Content $log -Encoding Default | Write-Host }
    if ($p.ExitCode -ne 0) { Write-Host "Не удалось создать базу (код $($p.ExitCode))." -ForegroundColor Red; exit 1 }
}

$Log = Join-Path $Root "build\onec_designer.log"
if ($Dump) {
    if (-not (Test-Path $EpfPath)) { Write-Host "Нет файла $EpfPath" -ForegroundColor Red; exit 1 }
    $code = Invoke-Designer @("/F", "`"$Ib`"", "/DumpExternalDataProcessorOrReportToFiles", "`"$Src`"", "`"$EpfPath`"") $Log
} else {
    if (Test-Path $EpfPath) { Remove-Item $EpfPath }
    $code = Invoke-Designer @("/F", "`"$Ib`"", "/LoadExternalDataProcessorOrReportFromFiles", "`"$Src`"", "`"$EpfPath`"") $Log
    if ($code -eq 0 -and -not (Test-Path $EpfPath)) { $code = 1 }
}

if ($code -ne 0) { Write-Host "Ошибка (код $code), журнал выше и в $Log" -ForegroundColor Red; exit $code }
if ($Dump) { Write-Host "Исходники обновлены: onec\src" -ForegroundColor Green }
else { Write-Host "Готово: $EpfPath" -ForegroundColor Green }
exit 0
