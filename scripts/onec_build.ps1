# Сборка внешних обработок 1С из исходников onec\src в build\<имя>.epf (P15, P16; D54, D55).
#
#   .\scripts\onec_build.ps1                    # собрать все: DocaiCheck и DocaiLoad
#   .\scripts\onec_build.ps1 -Name DocaiLoad    # собрать одну
#   .\scripts\onec_build.ps1 -Name DocaiLoad -Dump
#                                   # обратно: выгрузить build\DocaiLoad.epf в onec\src
#                                   # (после правок обработки в Конфигураторе)
#
# Платформа — самая новая установленная (Find-Platform). Учебной Бухгалтерии 3.0.111 нужна
# 8.3.16+, поэтому с P16 это учебная 8.3.18; исходники остаются в формате выгрузки 2.1.
#
# DocaiLoad ссылается на объекты Бухгалтерии (СправочникСсылка.Контрагенты и др.), поэтому
# собирается на базе с этой конфигурацией: -BpInfobase, по умолчанию пустая учебная база
# %USERPROFILE%\Documents\1c\docai_bp_empty (D58). База не должна быть открыта в 1С.
# Конфигуратор в пакетном режиме требует информационную базу. Скрипт создаёт пустую
# служебную базу build\onec_ib и работает только с ней: ваши базы не открываются.
param(
    [string[]]$Name = @("DocaiCheck", "DocaiLoad"),
    [switch]$Dump,
    [string]$Platform = "",
    [string]$BpInfobase = (Join-Path $env:USERPROFILE "Documents\1c\docai_bp_empty")
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

# Служебная пустая база: нужна Конфигуратору, данные в ней не хранятся.
if (-not (Test-Path (Join-Path $Ib "1Cv8.1CD"))) {
    Write-Host "Создаю служебную базу $Ib"
    $log = Join-Path $Root "build\onec_create.log"
    if (Test-Path $log) { Remove-Item $log }
    $p = Start-Process -FilePath $Platform -ArgumentList @("CREATEINFOBASE", "File=`"$Ib`"", "/DisableStartupDialogs", "/Out", "`"$log`"") -Wait -PassThru -WindowStyle Hidden
    if (Test-Path $log) { Get-Content $log -Encoding Default | Write-Host }
    if ($p.ExitCode -ne 0) { Write-Host "Не удалось создать базу (код $($p.ExitCode))." -ForegroundColor Red; exit 1 }
}

if ($Dump -and $Name.Count -ne 1) { Write-Host "Для -Dump укажите одну обработку: -Name DocaiLoad" -ForegroundColor Red; exit 1 }

$Log = Join-Path $Root "build\onec_designer.log"
foreach ($item in $Name) {
    $Src = Join-Path $Root "onec\src\$item.xml"
    $EpfPath = Join-Path $Root "build\$item.epf"
    $Base = $Ib
    if ($item -eq "DocaiLoad") {
        if (-not (Test-Path (Join-Path $BpInfobase "1Cv8.1CD"))) {
            Write-Host "DocaiLoad собирается на базе учебной Бухгалтерии, а её нет: $BpInfobase. Укажите -BpInfobase <папка базы>." -ForegroundColor Red
            exit 1
        }
        $Base = $BpInfobase
    }
    if (-not (Test-Path $Src)) { Write-Host "Нет исходников $Src" -ForegroundColor Red; exit 1 }
    if ($Dump) {
        if (-not (Test-Path $EpfPath)) { Write-Host "Нет файла $EpfPath" -ForegroundColor Red; exit 1 }
        $code = Invoke-Designer @("/F", "`"$Base`"", "/DumpExternalDataProcessorOrReportToFiles", "`"$Src`"", "`"$EpfPath`"") $Log
    } else {
        if (Test-Path $EpfPath) { Remove-Item $EpfPath }
        $code = Invoke-Designer @("/F", "`"$Base`"", "/LoadExternalDataProcessorOrReportFromFiles", "`"$Src`"", "`"$EpfPath`"") $Log
        if ($code -eq 0 -and -not (Test-Path $EpfPath)) { $code = 1 }
    }
    if ($code -ne 0) { Write-Host "Ошибка ($item, код $code), журнал выше и в $Log" -ForegroundColor Red; exit $code }
    if ($Dump) { Write-Host "Исходники обновлены: onec\src\$item" -ForegroundColor Green }
    else { Write-Host "Готово: $EpfPath" -ForegroundColor Green }
}
exit 0
