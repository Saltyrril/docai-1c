# Подключение дообученной модели к Ollama (шаг P21, docs/FINETUNE.md, шаг 5).
# Запуск из корня репозитория:
#   .\scripts\finetune_install.ps1                       # файлы в data\models\docai-qwen35-2b-ft
#   .\scripts\finetune_install.ps1 -Dir "$env:USERPROFILE\Downloads\docai-qwen35-2b-ft"
#   .\scripts\finetune_install.ps1 -UseBaseProjector     # проектор картинок от qwen3.5:2b-q8_0 (запасной путь)
# Делает: проверяет файлы (SHA-256 из export_info.json) → пишет Modelfile с полными путями →
# ollama create → проверяет, что у модели есть vision → docai finetune check-model (5 документов val).
param(
    [string]$Dir = "data\models\docai-qwen35-2b-ft",
    [string]$Name = "docai-qwen35-2b-ft",
    [switch]$UseBaseProjector,
    [string]$BaseModel = "qwen3.5:2b-q8_0",
    [switch]$SkipHash,
    [switch]$SkipCheck,
    [string]$OllamaUrl = "http://127.0.0.1:11434"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location (Split-Path -Parent $PSScriptRoot)

# Внешние программы (ollama, uv) пишут прогресс в stderr; Windows PowerShell 5.1 при
# $ErrorActionPreference = "Stop" и перенаправлении вывода считает это ошибкой. Поэтому их
# запускаем через Invoke-Native, а успех проверяем по коду выхода.
function Invoke-Native([scriptblock]$Block) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Block } finally { $ErrorActionPreference = $previous }
}

function Fail([string]$Message) {
    Write-Host $Message -ForegroundColor Red
    exit 1
}

# 1. Ollama должна быть запущена.
try {
    $version = Invoke-RestMethod -Uri "$OllamaUrl/api/version" -TimeoutSec 3
    Write-Host "Ollama работает, версия $($version.version)"
} catch {
    Fail "Ollama не отвечает на $OllamaUrl. Запустите приложение Ollama и повторите."
}

# 2. Файлы из Google Drive.
$dirPath = (Resolve-Path -LiteralPath $Dir -ErrorAction SilentlyContinue)
if (-not $dirPath) { Fail "Нет папки $Dir. Скачайте папку export\$Name из Google Drive (docs/FINETUNE.md, шаг 5)." }
$dirPath = $dirPath.Path
$gguf = Join-Path $dirPath "$Name.Q8_0.gguf"
$mmproj = Join-Path $dirPath "$Name.mmproj-f16.gguf"
$infoPath = Join-Path $dirPath "export_info.json"
if (-not (Test-Path -LiteralPath $gguf)) { Fail "Нет файла $gguf" }
$info = $null
if (Test-Path -LiteralPath $infoPath) {
    $info = Get-Content -LiteralPath $infoPath -Raw -Encoding UTF8 | ConvertFrom-Json
} else {
    Write-Host "Предупреждение: нет export_info.json — файлы не проверить по SHA-256." -ForegroundColor Yellow
}

if ($UseBaseProjector) {
    # Энкодер картинок при обучении не менялся (finetune_vision_layers: false), поэтому проектор
    # базовой модели подходит байт в байт по смыслу. Если энкодер обучали — так нельзя.
    if ($info -and $info.train_config.lora.finetune_vision_layers) {
        Fail "Модель обучалась с finetune_vision_layers: true — проектор базовой модели к ней не подходит."
    }
    $baseModelfile = Invoke-Native { ollama show --modelfile $BaseModel 2>$null }
    if ($LASTEXITCODE -ne 0) { Fail "Нет базовой модели ${BaseModel}: ollama pull ${BaseModel}" }
    $froms = @($baseModelfile | Where-Object { $_ -match '^FROM ' } | ForEach-Object { $_.Substring(5).Trim() })
    if ($froms.Count -lt 2) { Fail "У $BaseModel не найден проектор картинок (второй FROM)." }
    $mmproj = $froms[1]
    Write-Host "Проектор картинок — от ${BaseModel}: $mmproj"
} elseif (-not (Test-Path -LiteralPath $mmproj)) {
    Fail "Нет файла $mmproj (или запустите с -UseBaseProjector)."
}

# 3. Целостность после скачивания.
if ($info -and $info.sha256 -and -not $SkipHash) {
    $toCheck = @("$Name.Q8_0.gguf")
    if (-not $UseBaseProjector) { $toCheck += "$Name.mmproj-f16.gguf" }
    foreach ($file in $toCheck) {
        $expected = $info.sha256.$file
        if (-not $expected) { continue }
        Write-Host "SHA-256 $file ..."
        $actual = (Get-FileHash -LiteralPath (Join-Path $dirPath $file) -Algorithm SHA256).Hash.ToLower()
        if ($actual -ne $expected) { Fail "$file повреждён при скачивании (SHA-256 не совпал). Скачайте заново." }
    }
    Write-Host "Файлы целы."
}

# 4. Modelfile с полными путями (параметры — как в notebooks/Modelfile.docai-qwen35-2b-ft).
$modelfile = Join-Path $dirPath "Modelfile.local"
$lines = @(
    "FROM $gguf",
    "FROM $mmproj",
    "TEMPLATE {{ .Prompt }}",
    "RENDERER qwen3.5",
    "PARSER qwen3.5",
    "PARAMETER temperature 0",
    "PARAMETER top_k 20",
    "PARAMETER top_p 0.95",
    "PARAMETER presence_penalty 0",
    "PARAMETER num_ctx 8192",
    'LICENSE """Apache-2.0: базовая модель Qwen3.5-2B, https://huggingface.co/Qwen/Qwen3.5-2B"""'
)
[System.IO.File]::WriteAllLines($modelfile, $lines, (New-Object System.Text.UTF8Encoding($false)))

Write-Host "ollama create $Name ..."
Invoke-Native { ollama create $Name -f $modelfile 2>&1 | Where-Object { $_ -notmatch 'gathering model components' } | ForEach-Object { "$_" } }
if ($LASTEXITCODE -ne 0) { Fail "ollama create не удался (см. сообщение выше и docs/FINETUNE.md, «Если что-то пошло не так»)." }

# 5. Модель должна уметь картинки: иначе проектор не подключился.
$show = Invoke-RestMethod -Uri "$OllamaUrl/api/show" -Method Post -ContentType "application/json" `
    -Body (@{ model = $Name } | ConvertTo-Json)
$caps = @($show.capabilities)
Write-Host "Возможности ${Name}: $($caps -join ', ')"
if ($caps -notcontains "vision") { Fail "У модели нет vision: проектор картинок не подключился." }

# 6. Проверка: 5 документов val (2 PDF по тексту, картинки clean/scan/photo) → валидный JSON.
if ($SkipCheck) { exit 0 }
$report = "reports\2026-11-20_p21_finetune\check_model.json"
Invoke-Native { uv run docai finetune check-model --model $Name --out $report }
exit $LASTEXITCODE
