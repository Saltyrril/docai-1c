# Общие функции скриптов onec_*.ps1 (подключается через точку: . .\onec_common.ps1).

function Find-Platform {
    # Учебная платформа (1cv8t.exe) или обычная (1cv8.exe), самая новая версия.
    $candidates = @()
    foreach ($base in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if (-not $base) { continue }
        foreach ($pair in @(@("1cv8t", "1cv8t.exe"), @("1cv8", "1cv8.exe"))) {
            $dir = Join-Path $base $pair[0]
            if (Test-Path $dir) {
                $candidates += Get-ChildItem $dir -Directory |
                    Where-Object { $_.Name -match '^\d+\.\d+\.\d+\.\d+$' } |
                    ForEach-Object { Join-Path $_.FullName ("bin\" + $pair[1]) } |
                    Where-Object { Test-Path $_ }
            }
        }
    }
    $candidates | Sort-Object { [version](Split-Path (Split-Path (Split-Path $_)) -Leaf) } -Descending |
        Select-Object -First 1
}
