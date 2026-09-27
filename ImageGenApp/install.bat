@echo off
setlocal EnableDelayedExpansion

:: -----------------------------------------------------------------------------
:: ImageGen Studio - install the Python packages (run once; safe to re-run)
:: Uses the portable Python in ..\python-3.10 (installer\setup.bat creates it).
:: -----------------------------------------------------------------------------

set "APP_DIR=%~dp0"
set "REPO_DIR=%APP_DIR%..\"
set "PYTHON_DIR=%REPO_DIR%python-3.10"
set "PYTHON=%PYTHON_DIR%\python.exe"
set "NO_PAUSE="
if /i "%~1"=="--no-pause" set "NO_PAUSE=1"

cd /d "%APP_DIR%"

if not exist "%PYTHON%" (
    echo [ERROR] Portable Python not found at: !PYTHON!
    echo         Run installer\setup.bat first - it downloads Python 3.10 and everything else.
    goto :fail
)

echo ======================================================
echo  ImageGen Studio - Installing Python packages
echo ======================================================
echo  Python : !PYTHON!
echo ======================================================

echo.
echo [1/3] Upgrading pip...
"%PYTHON%" -m pip install --upgrade pip --quiet --no-warn-script-location

:: PyTorch CUDA 11.8 build: ZLUDA runs CUDA programs on AMD GPUs
echo.
echo [2/3] Installing PyTorch 2.4.1 ^(CUDA 11.8 build, ~2.5 GB download^)...
"%PYTHON%" -m pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 ^
    --index-url https://download.pytorch.org/whl/cu118 --no-warn-script-location
if !ERRORLEVEL! neq 0 (
    echo [ERROR] PyTorch could not be installed - check your internet connection and re-run.
    goto :fail
)

echo.
echo [3/3] Installing diffusers, gradio and the rest ^(requirements.txt^)...
"%PYTHON%" -m pip install -r "%APP_DIR%requirements.txt" --no-warn-script-location
if !ERRORLEVEL! neq 0 (
    echo [ERROR] Some packages failed to install - see the messages above, then re-run.
    goto :fail
)

:: -- HIP runtime check (ZLUDA needs AMD's HIP SDK 6.x, or the bundled therock_sdk) --
set "ROCM_BIN="
set "ROCM_ENV_FILE=%TEMP%\imagegen_rocm_%RANDOM%%RANDOM%.bat"
"%PYTHON%" "%APP_DIR%backend\rocm_env.py" --bat > "!ROCM_ENV_FILE!" 2>nul
if exist "!ROCM_ENV_FILE!" (
    call "!ROCM_ENV_FILE!"
    del "!ROCM_ENV_FILE!" 2>nul
)
echo.
if defined ROCM_BIN (
    echo [OK] AMD HIP runtime: !ROCM_RUNTIME! - GPU !ROCM_ARCH!
    if defined ROCM_NOTE echo [WARN] !ROCM_NOTE!
) else (
    echo [WARN] AMD HIP SDK not found. ZLUDA needs it to use your GPU.
    echo        Install "AMD HIP SDK for Windows" 6.4:
    echo        https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html
    if exist "!REPO_DIR!installer\hip_sdk_installer.exe" (
        choice /C YN /M "Run the included installer\hip_sdk_installer.exe now"
        if !ERRORLEVEL!==1 start "" /wait "!REPO_DIR!installer\hip_sdk_installer.exe"
    )
)

echo.
echo ======================================================
echo  Installation complete. Start the app with launch.bat
echo  Optional: install_npu.bat adds Ryzen AI NPU support.
echo ======================================================
if not "%NO_PAUSE%"=="1" pause
exit /b 0

:fail
if not "%NO_PAUSE%"=="1" pause
exit /b 1
