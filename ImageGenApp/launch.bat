@echo off
setlocal EnableDelayedExpansion

:: -----------------------------------------------------------------------------
:: ImageGen Studio - Windows launcher with ZLUDA v6 (CUDA on AMD via HIP)
:: Runtime: bundled therock_sdk, else the system HIP SDK 6.x; gfx1031 GPUs also
:: get the bundled gfx1031 rocBLAS kernels (see backend\rocm_env.py).
:: Note: cmd.exe can't handle a "!" or "%" in the install folder's path.
:: -----------------------------------------------------------------------------

set "APP_DIR=%~dp0"
set "REPO_DIR=%APP_DIR%..\"
set "PYTHON_DIR=%REPO_DIR%python-3.10"
set "PYTHON=%PYTHON_DIR%\python.exe"

:: -- Parse command-line flags (first: the GPU choice decides the runtime) -------
set "EXTRA_ARGS="
set "PORT=7860"
set "GPU_INDEX="
set "NO_ZLUDA=0"
set "NO_PAUSE="

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--help" goto :usage
if /i "%~1"=="-h"     goto :usage
if /i "%~1"=="/?"     goto :usage
if /i "%~1"=="--port" (
    set "ARGVAL=%~2"
    if not defined ARGVAL ( echo [ERROR] --port needs a number, e.g. --port 7861 & goto :fail )
    if "!ARGVAL:~0,1!"=="-" ( echo [ERROR] --port needs a number, e.g. --port 7861 & goto :fail )
    set "PORT=!ARGVAL!" & shift & shift & goto :parse_args
)
if /i "%~1"=="--gpu" (
    set "ARGVAL=%~2"
    if not defined ARGVAL ( echo [ERROR] --gpu needs a device number, e.g. --gpu 1 & goto :fail )
    if "!ARGVAL:~0,1!"=="-" ( echo [ERROR] --gpu needs a device number, e.g. --gpu 1 & goto :fail )
    set "GPU_INDEX=!ARGVAL!" & shift & shift & goto :parse_args
)
if /i "%~1"=="--share"      ( set "EXTRA_ARGS=!EXTRA_ARGS! --share" & shift & goto :parse_args )
if /i "%~1"=="--no-browser" ( set "EXTRA_ARGS=!EXTRA_ARGS! --no-browser" & shift & goto :parse_args )
if /i "%~1"=="--no-pause"   ( set "NO_PAUSE=1" & shift & goto :parse_args )
if /i "%~1"=="--cpu"        ( set "FORCE_CPU=1" & set "NO_ZLUDA=1" & shift & goto :parse_args )
if /i "%~1"=="--nozluda"    ( set "NO_ZLUDA=1" & set "IMAGEGEN_BACKEND=directml" & shift & goto :parse_args )
if /i "%~1"=="--dml"        ( set "NO_ZLUDA=1" & set "IMAGEGEN_BACKEND=directml" & shift & goto :parse_args )
if /i "%~1"=="--directml"   ( set "NO_ZLUDA=1" & set "IMAGEGEN_BACKEND=directml" & shift & goto :parse_args )
if /i "%~1"=="--zluda"      ( set "NO_ZLUDA=0" & set "IMAGEGEN_BACKEND=cuda" & shift & goto :parse_args )
echo [WARN] Unknown option "%~1" ignored ^(launch.bat --help lists them^)
shift & goto :parse_args

:usage
echo Usage: launch.bat [options]
echo   --port N       web UI port ^(default 7860; a busy port moves to the next free one^)
echo   --gpu N        HIP device number to use ^(default: the best AMD GPU^)
echo   --no-browser   don't open the browser
echo   --share        also create a public gradio.live link
echo   --cpu          run on the CPU ^(very slow^)
echo   --dml          use DirectML instead of ZLUDA
echo   --no-pause     don't wait for a key press when the app exits
exit /b 0

:args_done
:: -- ZLUDA v6 ----------------------------------------------------------------
set "ZLUDA_DIR=%REPO_DIR%ZLUDA_v6\zluda"
set "ZLUDA_EXE=%REPO_DIR%ZLUDA_v6\zluda\zluda.exe"
set "USE_ZLUDA=0"
if "!NO_ZLUDA!"=="0" (
    if exist "!ZLUDA_EXE!" (
        set "USE_ZLUDA=1"
    ) else (
        echo [WARN] ZLUDA not found in !ZLUDA_DIR! - GPU acceleration is off
    )
)

:: -- Change to app directory so Python can find config.py / backend/ -----------
cd /d "%APP_DIR%"

:: -- Clear stale bytecode cache (prevents old .pyc overriding edited .py) -----
if exist "%APP_DIR%__pycache__" rd /s /q "%APP_DIR%__pycache__" 2>nul
if exist "%APP_DIR%backend\__pycache__" rd /s /q "%APP_DIR%backend\__pycache__" 2>nul

:: -- Verify bundled Python -----------------------------------------------------
if not exist "%PYTHON%" (
    echo [WARN] Bundled Python not found, falling back to system python
    set "PYTHON=python"
)

:: -- Pick the GPU, HIP runtime and rocBLAS kernels ---------------------------------
:: backend\rocm_env.py asks hipInfo which AMD GPUs exist (the best one unless --gpu
:: is given) and picks the runtime that has kernels for it: therock_sdk (bundled,
:: RDNA 1/2) or the system HIP SDK 6.x (RDNA 3/4, or when therock_sdk is absent).
:: gfx1031 (RX 6700/6750 XT, 6700M/6800M/6850M XT) also gets the bundled gfx1031
:: rocBLAS kernels - other GPUs must not.
set "HIP_FOUND=0"
set "ROCM_BIN="
set "ROCM_NOTE="
set "ROCM_GPU_INDEX="
set "IMAGEGEN_GPU=!GPU_INDEX!"
rem A fresh file per launch: a leftover one from another run must never be used
set "ROCM_ENV_FILE=%TEMP%\imagegen_rocm_%RANDOM%%RANDOM%.bat"
if exist "%PYTHON%" (
    "%PYTHON%" "%APP_DIR%backend\rocm_env.py" --bat > "!ROCM_ENV_FILE!" 2>nul
    if exist "!ROCM_ENV_FILE!" (
        call "!ROCM_ENV_FILE!"
        del "!ROCM_ENV_FILE!" 2>nul
    )
)
if not defined GPU_INDEX set "GPU_INDEX=!ROCM_GPU_INDEX!"
if not defined GPU_INDEX set "GPU_INDEX=0"
if defined ROCM_BIN (
    set "HIP_FOUND=1"
    echo [INFO] GPU !GPU_INDEX! !ROCM_ARCH! - ROCm runtime: !ROCM_RUNTIME! ^(!ROCM_BIN!^)
    if defined ROCM_NOTE echo [WARN] !ROCM_NOTE!
) else if "!USE_ZLUDA!"=="1" (
    echo.
    echo [WARN] No usable AMD HIP runtime found ^(bundled therock_sdk or HIP SDK 6.x^).
    echo.
    echo  To fix this, install the AMD HIP SDK 6.4 ^(installer\hip_sdk_installer.exe if
    echo  your copy includes it, or https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html^)
    echo  and re-run this launcher. Continuing without ZLUDA GPU support...
    echo.
    set "USE_ZLUDA=0"
)

:: -- ROCm first, then the ZLUDA dir (its nvcuda.dll intercepts CUDA calls) ------
if "!USE_ZLUDA!"=="1" set "PATH=!ROCM_BIN!;!ZLUDA_DIR!;!PATH!"

:: -- ZLUDA / HIP environment flags --------------------------------------------
set "ZLUDA_NO_TELEMETRY=1"
:: Required: hipBLASLt has no gfx103x kernels, so every Linear+bias would fail
set "DISABLE_ADDMM_CUDA_LT=1"
set "MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK=0"
:: Suppress MIOpen gfx908 AI heuristic warnings (non-fatal, falls back to generic)
set "MIOPEN_LOG_LEVEL=0"
set "MIOPEN_ENABLE_LOGGING=0"
set "MIOPEN_ENABLE_LOGGING_CMD=0"
set "GRADIO_ANALYTICS_ENABLED=False"
set "HF_HUB_OFFLINE=0"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "GRADIO_SERVER_NAME=127.0.0.1"

if exist "!REPO_DIR!.hf_cache" (
    set "HF_HOME=!REPO_DIR!.hf_cache"
    set "HUGGINGFACE_HUB_CACHE=!REPO_DIR!.hf_cache\hub"
)
if exist "!REPO_DIR!.miopen_cache" (
    set "MIOPEN_CACHE_DIR=!REPO_DIR!.miopen_cache"
)
if exist "!REPO_DIR!.miopen_userdb" (
    set "MIOPEN_USER_DB_PATH=!REPO_DIR!.miopen_userdb"
)

:: Quick sanity check: can Python import the app at all?
"%PYTHON%" -c "import sys; sys.path.insert(0,'.'); import config" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Python cannot import core modules. Dependencies may be missing.
    echo  Run install.bat first, then re-run this launcher.
    echo.
    goto :fail
)

:: ZLUDA v6: do NOT set HIP_VISIBLE_DEVICES - it breaks multi-GPU detection.
:: IMAGEGEN_GPU makes the app pick the matching torch.cuda device instead.
set "IMAGEGEN_GPU=!GPU_INDEX!"

set "ZLUDA_SHOW=off"
if "!USE_ZLUDA!"=="1" set "ZLUDA_SHOW=v6 (!ZLUDA_DIR!)"
echo ======================================================
echo  ImageGen Studio
echo ======================================================
echo  Python : !PYTHON!
echo  ZLUDA  : !ZLUDA_SHOW!
echo  ROCm   : !ROCM_BIN!
echo  GPU    : device !GPU_INDEX!
echo  Port   : !PORT!
echo ======================================================

:launch
if "!USE_ZLUDA!"=="1" (
    echo [INFO] Launching with ZLUDA v6...
    "!ZLUDA_EXE!" -- "!PYTHON!" app.py --port !PORT! !EXTRA_ARGS!
) else (
    echo [INFO] Launching without ZLUDA ^(CPU or DirectML only^)...
    "!PYTHON!" app.py --port !PORT! !EXTRA_ARGS!
)
set "APP_EXIT=!ERRORLEVEL!"

if not "!APP_EXIT!"=="0" (
    echo.
    echo [ERROR] App exited with code !APP_EXIT!
    if "!HIP_FOUND!"=="0" (
        echo.
        echo  The most likely cause is missing AMD ROCm DLLs - install the AMD HIP SDK 6.4
        echo  ^(installer\hip_sdk_installer.exe if included^), then re-run this launcher.
    ) else (
        echo  Run install.bat first if Python dependencies are missing.
    )
) else (
    echo.
    echo [INFO] App exited normally.
)
echo.
if not "%NO_PAUSE%"=="1" pause
exit /b !APP_EXIT!

:fail
if not "%NO_PAUSE%"=="1" pause
exit /b 1
