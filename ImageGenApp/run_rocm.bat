@echo off
setlocal EnableDelayedExpansion

rem Usage: run_rocm.bat script.py [args]  -- the native ROCm environment launch.bat uses (install_rocm.bat)

set "APP_DIR=%~dp0"
set "REPO_DIR=%APP_DIR%..\"
set "PYTHON=%REPO_DIR%python-rocm\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo [ERROR] Native ROCm environment not found: run install_rocm.bat first.
    exit /b 1
)

set "ROCM_ENV_FILE=%TEMP%\imagegen_rocm_%RANDOM%%RANDOM%.bat"
"%PYTHON%" "%APP_DIR%backend\rocm_env.py" --bat > "%ROCM_ENV_FILE%" 2>nul
if exist "%ROCM_ENV_FILE%" call "%ROCM_ENV_FILE%"
del "%ROCM_ENV_FILE%" 2>nul
if not defined IMAGEGEN_GPU set "IMAGEGEN_GPU=!ROCM_GPU_INDEX!"
if not defined IMAGEGEN_GPU set "IMAGEGEN_GPU=0"
set "ROCBLAS_TENSILE_LIBPATH="

set "GRADIO_ANALYTICS_ENABLED=False"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
if exist "%REPO_DIR%.hf_cache" (
    set "HF_HOME=%REPO_DIR%.hf_cache"
    set "HUGGINGFACE_HUB_CACHE=%REPO_DIR%.hf_cache\hub"
)

cd /d "%APP_DIR%"
"%PYTHON%" %*
