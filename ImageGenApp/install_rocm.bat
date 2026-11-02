@echo off
setlocal EnableDelayedExpansion

:: -----------------------------------------------------------------------------
:: ImageGen Studio - native ROCm PyTorch (no ZLUDA) in ..\python-rocm
:: Usage: install_rocm.bat [gfx1201 gfx1103 ...] [--no-pause]
:: Without architectures it asks the GPUs that are in this PC. launch.bat then
:: uses this environment for those GPUs by itself; launch.bat --zluda keeps ZLUDA.
:: Needs Python 3.12 (the py launcher) and a current AMD Adrenalin driver.
:: -----------------------------------------------------------------------------

set "APP_DIR=%~dp0"
set "ENV_DIR=%APP_DIR%..\python-rocm"
set "INDEX=https://stable.repo.amd.com/rocm/whl-next/"
set "ROCM_VER=10.1.0"
set "ARCHS="
set "NO_PAUSE="

:parse
if "%~1"=="" goto :parsed
if /i "%~1"=="--no-pause" ( set "NO_PAUSE=1" ) else ( set "ARCHS=!ARCHS! %~1" )
shift & goto :parse
:parsed

if not exist "%ENV_DIR%\Scripts\python.exe" (
    echo [1/4] Creating the Python 3.12 environment...
    py -3.12 -m venv "%ENV_DIR%"
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Python 3.12 not found. Install it first: winget install Python.Python.3.12
        goto :fail
    )
)
set "PYTHON=%ENV_DIR%\Scripts\python.exe"

if not defined ARCHS for /f "usebackq delims=" %%A in (`call "%PYTHON%" "%APP_DIR%backend\rocm_env.py" --archs 2^>nul`) do set "ARCHS=%%A"
if not defined ARCHS (
    echo [ERROR] No AMD GPU found. Name it yourself, for example: install_rocm.bat gfx1201
    goto :fail
)

:: torch[device-gfx1201,device-gfx1103]: one environment can hold several GPUs' kernels
set "EXTRAS="
for %%A in (!ARCHS!) do (
    if defined EXTRAS ( set "EXTRAS=!EXTRAS!,device-%%A" ) else ( set "EXTRAS=device-%%A" )
)
echo [2/4] Installing PyTorch for:!ARCHS! ^(several GB^)...
"%PYTHON%" -m pip install --disable-pip-version-check --index-url %INDEX% "torch[!EXTRAS!]==2.14.0+rocm%ROCM_VER%" "torchvision[!EXTRAS!]==0.29.0a0+rocm%ROCM_VER%" "torchaudio==2.11.0.3+rocm%ROCM_VER%"
if !ERRORLEVEL! neq 0 (
    echo [ERROR] PyTorch could not be installed - check the architecture name and your internet connection.
    goto :fail
)

echo [3/4] Installing diffusers, gradio and the rest ^(requirements-rocm.txt^)...
:: torch stays pinned here: left free, pip's resolver swapped it for a CPU-only build from PyPI (torchsde)
"%PYTHON%" -m pip install --disable-pip-version-check --extra-index-url %INDEX% -r "%APP_DIR%requirements-rocm.txt" "torch==2.14.0+rocm%ROCM_VER%" "torchvision==0.29.0a0+rocm%ROCM_VER%" "torchaudio==2.11.0.3+rocm%ROCM_VER%"
if !ERRORLEVEL! neq 0 ( echo [ERROR] Some packages failed to install. & goto :fail )

echo [4/4] Checking the GPU...
"%PYTHON%" -c "import torch; print('GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]); x = torch.randn(512, 512, device='cuda'); print('matmul OK', float((x @ x).mean()) == float((x @ x).mean()))"
if !ERRORLEVEL! neq 0 ( echo [ERROR] PyTorch cannot use the GPU - update the Adrenalin driver. & goto :fail )

echo.
echo Done. launch.bat now starts on native ROCm for:!ARCHS!
if not "%NO_PAUSE%"=="1" pause
exit /b 0

:fail
if not "%NO_PAUSE%"=="1" pause
exit /b 1
