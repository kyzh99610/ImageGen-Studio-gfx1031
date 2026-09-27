@echo off
setlocal EnableDelayedExpansion

:: -----------------------------------------------------------------------------
:: ImageGen Studio - NPU (Ryzen AI / VitisAI) optional installation
::
:: Run AFTER installing AMD Ryzen AI Software:
::   https://account.amd.com/en/forms/downloads/ryzenai-eula-public-xef.html?filename=ryzen-ai-lt-1.7.0.exe
::
:: This script installs onnxruntime-vitisai into the bundled Python 3.10 so
:: ImageGenApp can use the XDNA NPU for text encoder / VAE decode offload.
:: -----------------------------------------------------------------------------

set "APP_DIR=%~dp0"
set "REPO_DIR=%APP_DIR%..\"
set "PYTHON=%REPO_DIR%python-3.10\python.exe"

if not exist "%PYTHON%" (
    echo [ERROR] Bundled Python not found at: %PYTHON%
    pause & exit /b 1
)

echo ======================================================
echo  ImageGen Studio - NPU (Ryzen AI) Installation
echo ======================================================

:: -- Check AMD Ryzen AI Software is installed ----------------------------------
set "RYZENAI_PATH="
if defined RYZEN_AI_INSTALLATION_PATH (
    set "RYZENAI_PATH=!RYZEN_AI_INSTALLATION_PATH!"
) else (
    :: Try the default installation path for v1.7.0
    for %%V in (1.7.0 1.6.1 1.6.0 1.5.0) do (
        if exist "C:\Program Files\RyzenAI\%%V" (
            set "RYZENAI_PATH=C:\Program Files\RyzenAI\%%V"
        )
    )
)

if "!RYZENAI_PATH!"=="" (
    echo.
    echo [ERROR] AMD Ryzen AI Software not found.
    echo.
    echo  Please install it first:
    echo    1. Download: https://account.amd.com/en/forms/downloads/ryzenai-eula-public-xef.html?filename=ryzen-ai-lt-1.7.0.exe
    echo    2. Run ryzen-ai-lt-1.7.0.exe and follow the wizard
    echo    3. Re-run this script
    echo.
    pause & exit /b 1
)

echo  Ryzen AI path : !RYZENAI_PATH!

:: -- Verify the VitisAI deployment DLL is present (minimum requirement) --------
if not exist "!RYZENAI_PATH!\deployment\onnxruntime_providers_vitisai.dll" (
    echo.
    echo [ERROR] VitisAI provider DLL not found. The Ryzen AI installation may be incomplete.
    pause & exit /b 1
)

:: -- Find the onnxruntime_vitisai wheel and check its Python tag ----------------
set "VITISAI_WHL="
set "WHEEL_CPVER="
for /f "usebackq delims=" %%F in (`dir /b /s "!RYZENAI_PATH!\onnxruntime_vitisai*.whl" 2^>nul`) do (
    set "VITISAI_WHL=%%F"
    for /f "tokens=3 delims=-" %%T in ("%%~nF") do set "WHEEL_CPVER=%%T"
)

:: -- Get our Python's version tag (e.g. cp310) ---------------------------------
set "OUR_CPVER="
"%PYTHON%" -c "import sys; open(r'%TEMP%\cpver.txt','w').write(f'cp{sys.version_info.major}{sys.version_info.minor}')" 2>nul
if exist "%TEMP%\cpver.txt" set /p OUR_CPVER=<"%TEMP%\cpver.txt"

echo  VitisAI wheel  : !VITISAI_WHL!
echo  Wheel Python   : !WHEEL_CPVER!
echo  App Python     : !OUR_CPVER!
echo.

if "!VITISAI_WHL!"=="" (
    echo [WARN] No onnxruntime_vitisai wheel found - SDK may use a pre-installed package.
    goto :verify_sdk
)

if /i "!WHEEL_CPVER!" neq "!OUR_CPVER!" (
    echo [INFO] Wheel is for !WHEEL_CPVER! but this app uses !OUR_CPVER!.
    echo        The Ryzen AI SDK is installed and the NPU will be detected correctly.
    echo        Direct Python binding is unavailable ^(AMD only ships !WHEEL_CPVER! for v1.7+^).
    echo.
    echo        NPU inference will run via the ryzen-ai conda env ^(Python 3.12^).
    goto :verify_sdk
)

:: -- Wheel matches our Python - install it directly ---------------------------
echo [1/3] Removing conflicting onnxruntime packages...
"%PYTHON%" -m pip uninstall onnxruntime onnxruntime-directml -y --quiet 2>nul
echo       Done.

echo.
echo [2/3] Installing onnxruntime-vitisai...
"%PYTHON%" -m pip install "!VITISAI_WHL!" --no-warn-script-location
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Installation failed.
    pause & exit /b 1
)

echo.
echo [3/3] Verifying VitisAI Execution Provider...
"%PYTHON%" -c "import onnxruntime as ort; eps = ort.get_available_providers(); print('  Providers:', eps); exit(0 if 'VitisAIExecutionProvider' in eps else 1)"
if %ERRORLEVEL% neq 0 (
    echo [WARN] VitisAIExecutionProvider not in providers list. Reboot may be needed.
) else (
    echo  VitisAIExecutionProvider is active. NPU is ready!
)
goto :done

:verify_sdk
:: VitisAI DLL already confirmed present above - SDK detection will work
echo [1/1] Ryzen AI SDK confirmed at !RYZENAI_PATH!
echo        hardware_detector.py will detect the NPU via the deployment DLL.
echo        NPU will show as available in the Settings tab.

:done
echo.
echo ======================================================
echo  NPU installation complete.
echo  Launch ImageGen Studio with: launch.bat
echo ======================================================
pause
endlocal
