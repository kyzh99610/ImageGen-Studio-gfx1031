@echo off
setlocal
:: -----------------------------------------------------------------------------
:: Studio Loop - render saved settings round after round, unattended, and sort
:: the pictures (see studio_loop.py). Plans live in settings\loop\<name>.json.
::   studio_loop.bat <plan>                          starts its own app
::   studio_loop.bat <plan> --attach http://127.0.0.1:7860
:: Stop it: create a file named STOP in the run folder, or close this window.
:: -----------------------------------------------------------------------------
set "APP_DIR=%~dp0"
set "PYTHON=%APP_DIR%..\python-3.10\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
if "%~1"=="" goto usage
"%PYTHON%" "%APP_DIR%studio_loop.py" %*
pause
exit /b
:usage
echo Usage: studio_loop.bat ^<plan^> [--attach URL]
echo Plans in "%APP_DIR%settings\loop":
if exist "%APP_DIR%settings\loop" dir /b "%APP_DIR%settings\loop\*.json"
echo Example plan: "%APP_DIR%loop_example.json"
pause
