@echo off
:: ImageGen Studio - Setup launcher
:: Double-click this on any new Windows PC to get fully set up.
:: Requires internet connection (~8 GB download on first run).

echo.
echo  Starting ImageGen Studio setup...
echo  This will install Python, packages, and optional models.
echo.

:: From a PowerShell 7 prompt this variable points Windows PowerShell at modules it can't load ("Get-FileHash is not recognized")
set "PSModulePath="
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*

if errorlevel 1 (
    echo.
    echo  Setup encountered an error. See messages above.
    pause
)
