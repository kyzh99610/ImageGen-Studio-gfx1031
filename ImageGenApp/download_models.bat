@echo off
setlocal

set "PYTHON=%~dp0..\python-3.10\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

echo ============================================================
echo  ImageGen Studio - Model Downloader
echo  Downloads ~6.3 GB of popular SD 1.5 models.
echo  Re-run anytime to resume interrupted downloads.
echo ============================================================
echo.

"%PYTHON%" "%~dp0download_models.py"

echo.
pause
