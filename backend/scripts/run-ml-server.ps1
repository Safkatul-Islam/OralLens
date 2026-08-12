$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$env:ORALLENS_INFERENCE_MODE = "ml"
$env:ORALLENS_ML_SOURCE_PATH = Join-Path $ProjectRoot "ml\src"
$env:ORALLENS_ML_DETECTION_CONFIG_PATH = Join-Path $ProjectRoot "ml\configs\orthodontic_plaque_detection_mvp_v3_predict.toml"
$env:ORALLENS_ML_TEMP_DIR = Join-Path $ProjectRoot "backend\var\ml-inputs"
Set-Location $ProjectRoot
& ".\ml\.venv\Scripts\python.exe" -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
