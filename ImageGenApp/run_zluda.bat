@echo off
setlocal EnableDelayedExpansion

rem Usage: run_zluda.bat script.py [args]  -- same ZLUDA environment as launch.bat
rem ZLUDA v6 + the HIP runtime backend\rocm_env.py picks for this GPU (therock_sdk or HIP SDK 6.x)

set "APP_DIR=%~dp0"
set "REPO_DIR=%APP_DIR%..\"
set "PYTHON_DIR=%REPO_DIR%python-3.10"
set "PYTHON=%PYTHON_DIR%\python.exe"

set "ZLUDA_EXE=%REPO_DIR%ZLUDA_v6\zluda\zluda.exe"
if not exist "%ZLUDA_EXE%" (
    echo [ERROR] ZLUDA v6 not found: !ZLUDA_EXE!
    exit /b 1
)

set "ROCM_BIN="
set "ROCM_ENV_FILE=%TEMP%\imagegen_rocm_%RANDOM%%RANDOM%.bat"
"%PYTHON%" "%APP_DIR%backend\rocm_env.py" --bat > "%ROCM_ENV_FILE%" 2>nul
if exist "%ROCM_ENV_FILE%" call "%ROCM_ENV_FILE%"
del "%ROCM_ENV_FILE%" 2>nul
if not defined ROCM_BIN (
    echo [ERROR] No usable HIP runtime ^(therock_sdk or AMD HIP SDK 6.x^).
    echo         Install the AMD HIP SDK 6.4 ^(installer\hip_sdk_installer.exe if included^).
    exit /b 1
)
if defined ROCM_NOTE echo [WARN] !ROCM_NOTE!
rem rocm_env.py used IMAGEGEN_GPU if set, else picked the best GPU: the app must use the same one
if not defined IMAGEGEN_GPU set "IMAGEGEN_GPU=!ROCM_GPU_INDEX!"
set "PATH=!ROCM_BIN!;%REPO_DIR%ZLUDA_v6\zluda;%PATH%"

set "ZLUDA_NO_TELEMETRY=1"
rem cublasLt -> hipBLASLt has no gfx103x kernels: every Linear-with-bias would fail
set "DISABLE_ADDMM_CUDA_LT=1"
set "MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK=0"
set "MIOPEN_LOG_LEVEL=0"
set "MIOPEN_ENABLE_LOGGING=0"
set "MIOPEN_ENABLE_LOGGING_CMD=0"
set "GRADIO_ANALYTICS_ENABLED=False"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
if not defined IMAGEGEN_GPU set "IMAGEGEN_GPU=0"

if exist "%REPO_DIR%.hf_cache" (
    set "HF_HOME=%REPO_DIR%.hf_cache"
    set "HUGGINGFACE_HUB_CACHE=%REPO_DIR%.hf_cache\hub"
)
if exist "%REPO_DIR%.miopen_cache" (
    set "MIOPEN_CACHE_DIR=%REPO_DIR%.miopen_cache"
)
if exist "%REPO_DIR%.miopen_userdb" (
    set "MIOPEN_USER_DB_PATH=%REPO_DIR%.miopen_userdb"
)

cd /d "%APP_DIR%"
"%ZLUDA_EXE%" -- "%PYTHON%" %*
