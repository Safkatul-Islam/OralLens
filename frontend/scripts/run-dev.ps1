$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
Set-Location (Join-Path $ProjectRoot "frontend")
& "C:\Program Files\nodejs\npm.cmd" run dev -- --host 127.0.0.1 --port 5173
