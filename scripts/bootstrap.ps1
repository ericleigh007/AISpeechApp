param(
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$ModelRoot = "D:\AI\Models"

$env:AI_SPEECH_MODEL_ROOT = $ModelRoot
$env:HF_HOME = Join-Path $ModelRoot "huggingface"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:TRANSFORMERS_CACHE = $env:HUGGINGFACE_HUB_CACHE
$env:HF_XET_CACHE = Join-Path $env:HF_HOME "xet"
$env:MODELSCOPE_CACHE = Join-Path $ModelRoot "modelscope"
$env:TORCH_HOME = Join-Path $ModelRoot "torch"

if (-not $Python) {
    $Candidates = @(
        "$env:APPDATA\uv\python\cpython-3.11.13-windows-x86_64-none\python.exe",
        "$env:APPDATA\uv\python\cpython-3.12.11-windows-x86_64-none\python.exe",
        "python"
    )

    foreach ($Candidate in $Candidates) {
        try {
            $Version = & $Candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
            if ($LASTEXITCODE -eq 0 -and [version]$Version -ge [version]"3.11" -and [version]$Version -lt [version]"3.14") {
                $Python = $Candidate
                break
            }
        } catch {
            continue
        }
    }
}

if (-not $Python) {
    throw "Could not find Python >=3.11 and <3.14. Pass -Python C:\path\to\python.exe."
}

Push-Location $Root
try {
    & $Python -m venv .venv
    & .\.venv\Scripts\python.exe -m pip install --upgrade pip
    & .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
    & .\.venv\Scripts\python.exe -m pytest -q
    & .\.venv\Scripts\python.exe -m aispeechapp.smoke --all --metadata-only
} finally {
    Pop-Location
}
