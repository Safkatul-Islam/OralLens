@echo off
set SCRIPT_DIR=%~dp0
for %%I in ("%SCRIPT_DIR%..\..") do set PROJECT_ROOT=%%~fI
set ORALLENS_INFERENCE_MODE=ml
set ORALLENS_ML_SOURCE_PATH=%PROJECT_ROOT%\ml\src
set ORALLENS_ML_DETECTION_CONFIG_PATH=%PROJECT_ROOT%\ml\configs\orthodontic_plaque_detection_mvp_predict.toml
set ORALLENS_ML_TEMP_DIR=%PROJECT_ROOT%\backend\var\ml-inputs
cd /d "%PROJECT_ROOT%"
ml\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
