$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Chưa có .venv. Xem các bước cài đặt trong README.md.'
}
Set-Location -LiteralPath $projectRoot
& $projectPython -m streamlit run app.py --server.address 127.0.0.1

