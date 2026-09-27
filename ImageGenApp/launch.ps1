#!/usr/bin/env pwsh
<#
.SYNOPSIS
  ImageGen Studio - PowerShell launcher with ZLUDA (AMD ROCm via CUDA emulation)
#>

param(
    [int]$Port = 7860,
    [int]$Gpu  = -1,   # -1 = the best AMD GPU (backend\rocm_env.py)
    [switch]$Share,
    [switch]$NoPause,
    [switch]$NoBrowser,
    [switch]$Cpu,
    [switch]$NoZluda,
    [switch]$Dml,
    [switch]$Zluda
)

$AppDir   = $PSScriptRoot
$RepoDir  = Split-Path $AppDir -Parent
$PyDir    = Join-Path $RepoDir "python-3.10"
$Python   = Join-Path $PyDir "python.exe"

# ── ZLUDA v6 (+ therock_sdk, or the system HIP SDK 6.x when therock_sdk is absent) ──
# The old ZLUDA v5 build was removed 2026-09-26: it failed every GPU op on this driver.
$ZludaVer = 0
$ZludaDir = Join-Path $RepoDir "ZLUDA_v6\zluda"
$ZludaExe = $null
if (Test-Path (Join-Path $ZludaDir "zluda.exe")) {
    $ZludaExe = Join-Path $ZludaDir "zluda.exe"
    $ZludaVer = 6
}

# ── Change to app directory (critical: lets Python find config.py) ────────────
Set-Location $AppDir

# ── Clear stale bytecode cache (prevents old .pyc overriding edited .py) ──────
Remove-Item "$AppDir\__pycache__" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$AppDir\backend\__pycache__" -Recurse -Force -ErrorAction SilentlyContinue

# ── Python ────────────────────────────────────────────────────────────────────
if (-not (Test-Path $Python)) {
    Write-Warning "Bundled Python not found, falling back to system python"
    $Python = "python"
}

# ── Pick the HIP runtime + rocBLAS kernels for this GPU (backend\rocm_env.py) ──
# hipInfo reports the GPU architecture; therock_sdk (bundled) has RDNA 1/2 kernels, the
# system HIP SDK 6.x has RDNA 3/4. gfx1031 also gets the bundled gfx1031 rocBLAS kernels.
$RocmBin = $null
$Rocm = @{}
# An explicit -Gpu decides the runtime too; otherwise rocm_env.py picks the best GPU
if ($Gpu -ge 0) { $env:IMAGEGEN_GPU = "$Gpu" } else { Remove-Item Env:IMAGEGEN_GPU -ErrorAction SilentlyContinue }
try {
    $json = & $Python (Join-Path $AppDir "backend\rocm_env.py") --json 2>$null
    if ($LASTEXITCODE -eq 0 -and $json) { $Rocm = ($json -join "`n") | ConvertFrom-Json }
} catch { }
if ($Rocm.rocm_bin) {
    $RocmBin = $Rocm.rocm_bin
    Write-Host "[INFO] GPU $($Rocm.gpu_index) $($Rocm.arch) - ROCm runtime: $($Rocm.runtime) ($RocmBin)"
    if ($Rocm.note) { Write-Warning $Rocm.note }
} else {
    Write-Warning @"
No usable AMD HIP runtime found (bundled therock_sdk or HIP SDK 6.x).

To fix this, install the AMD HIP SDK 6.4 (installer\hip_sdk_installer.exe if your copy
includes it, or https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)
and re-run this launcher. Continuing in CPU/DirectML mode.
"@
    $NoZluda = $true
}

# ── ZLUDA ─────────────────────────────────────────────────────────────────────
if ($Dml -or $NoZluda) {
    $env:IMAGEGEN_BACKEND = "directml"
}
if ($Zluda) {
    $env:IMAGEGEN_BACKEND = "cuda"
}
$UseZluda = (-not $NoZluda) -and (-not $Dml) -and (-not $Cpu) -and ($null -ne $ZludaExe) -and (Test-Path $ZludaExe) -and ($null -ne $RocmBin)
if ($UseZluda) {
    $env:ZLUDA_NO_TELEMETRY   = "1"
    $env:DISABLE_ADDMM_CUDA_LT = "1"  # cublasLt -> hipBLASLt has no gfx103x kernels (every Linear+bias fails)
    $env:MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK = "0"
    $env:MIOPEN_LOG_LEVEL = "0"       # suppress non-fatal gfx908 heuristic warnings
    $env:MIOPEN_ENABLE_LOGGING = "0"
    $env:MIOPEN_ENABLE_LOGGING_CMD = "0"

    $hfCache = Join-Path $RepoDir ".hf_cache"
    if (Test-Path $hfCache) {
        $env:HF_HOME = $hfCache
        $env:HUGGINGFACE_HUB_CACHE = Join-Path $hfCache "hub"
    }
    $miopenCache = Join-Path $RepoDir ".miopen_cache"
    if (Test-Path $miopenCache) {
        $env:MIOPEN_CACHE_DIR = $miopenCache
    }
    $miopenUserDb = Join-Path $RepoDir ".miopen_userdb"
    if (Test-Path $miopenUserDb) {
        $env:MIOPEN_USER_DB_PATH = $miopenUserDb
    }

    # ROCm first, then the ZLUDA dir (its nvcuda.dll intercepts CUDA calls)
    $env:PATH = "$RocmBin;$ZludaDir;$env:PATH"
    $env:HIP_PATH = $Rocm.hip_path
    $env:ROCM_ARCH = $Rocm.arch
    if ($Rocm.tensile) { $env:ROCBLAS_TENSILE_LIBPATH = $Rocm.tensile }
}

# ── Verify core Python dependencies before launch ─────────────────────────────
& $Python -c "import sys; sys.path.insert(0,'.'); import config" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Python cannot import core modules. Dependencies may be missing." -ForegroundColor Red
    Write-Host "  Run install.bat first, then re-run this launcher." -ForegroundColor Yellow
    if (-not $NoPause) { Read-Host "Press Enter to exit" }
    exit 1
}

# ── GPU: -Gpu N, else the one rocm_env.py picked (HIP order = ZLUDA's CUDA order) ──
$GpuIndex = if ($Gpu -ge 0) { $Gpu } elseif ($null -ne $Rocm.gpu_index) { [int]$Rocm.gpu_index } else { 0 }
# ZLUDA v6: do NOT set HIP_VISIBLE_DEVICES — it breaks multi-GPU detection.
# IMAGEGEN_GPU makes the app pick the matching torch.cuda device instead.
$env:IMAGEGEN_GPU = "$GpuIndex"
if ($Cpu) { $env:FORCE_CPU = "1" }

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  ImageGen Studio" -ForegroundColor White
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  Python : $Python"
Write-Host "  ROCm   : $(if($RocmBin){$RocmBin}else{'not found'})"
Write-Host "  ZLUDA  : v$ZludaVer $(if($UseZluda){'enabled'}else{'disabled (CPU/DirectML)'})"
Write-Host "  GPU    : device $GpuIndex"
Write-Host "  Port   : $Port"
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

$ExtraArgs = @("--port", $Port)
if ($Share) { $ExtraArgs += "--share" }
if ($NoBrowser) { $ExtraArgs += "--no-browser" }

try {
    if ($UseZluda) {
        Write-Host "[INFO] Launching with ZLUDA v6 (DLL injection)..." -ForegroundColor Green
        & $ZludaExe -- $Python "app.py" @ExtraArgs
    } else {
        Write-Host "[INFO] Launching without ZLUDA (CPU/DirectML mode)..." -ForegroundColor Yellow
        & $Python "app.py" @ExtraArgs
    }
} catch {
    Write-Error "Failed to start app: $_"
}

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] App exited with code $LASTEXITCODE" -ForegroundColor Red
    if (-not $RocmBin) {
        Write-Host "  Install the AMD HIP SDK 6.4 first (installer\hip_sdk_installer.exe if included)." -ForegroundColor Yellow
    } else {
        Write-Host "  Run install.bat if Python dependencies are missing." -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "[INFO] App exited normally." -ForegroundColor Green
}
if (-not $NoPause) { Read-Host "Press Enter to exit" }
