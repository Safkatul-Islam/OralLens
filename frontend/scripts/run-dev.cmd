@echo off
set SCRIPT_DIR=%~dp0
for %%I in ("%SCRIPT_DIR%..\..") do set PROJECT_ROOT=%%~fI
set VITE_API_BASE_URL=http://127.0.0.1:8000
cd /d "%PROJECT_ROOT%\frontend"
npm.cmd run dev -- --host 127.0.0.1 --port 5173
