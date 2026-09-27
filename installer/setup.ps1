<#
.SYNOPSIS
    ImageGen Studio - fresh-PC setup (run installer\setup.bat, or this script directly).

.DESCRIPTION
    Everything goes inside this folder; nothing is installed system-wide except the AMD
    HIP SDK, which AMD distributes itself (the script tells you when it is missing).

      1. Python 3.10.11 portable (python.org embeddable zip) + pip
      2. ZLUDA v6 (github.com/vosen/ZLUDA, release v6) -> ZLUDA_v6\zluda
      3. gfx1031 rocBLAS kernels (github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU,
         release v0.6.2.4) -> _gfx1031_rocm624\  (only used on gfx1031 GPUs)
      4. Python packages (ImageGenApp\install.bat -> requirements.txt, ~5 GB)
      5. AMD HIP SDK 6.x check (needed by ZLUDA)
      6. Optional starter models (~6.3 GB)

    Downloads are checked against the SHA-256 of the files this project was tested with.

.PARAMETER SkipModels   Don't offer the starter models.
.PARAMETER Unattended   No questions; use the defaults (models are skipped).
#>

[CmdletBinding()]
param(
    [switch]$SkipModels,
    [switch]$SkipRocm,
    [switch]$Unattended
)

Set-StrictMode -Off
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"      # Invoke-WebRequest is 10x slower with the progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$ROOT   = Split-Path $PSScriptRoot -Parent    # installer\ -> project root
$PYTHON = Join-Path $ROOT "python-3.10"
$PYEXE  = Join-Path $PYTHON "python.exe"
$APPDIR = Join-Path $ROOT "ImageGenApp"
$TMPDIR = Join-Path $ROOT "installer\tmp"

$ZLUDA_URL    = "https://github.com/vosen/ZLUDA/releases/download/v6/zluda-windows-3fe1206.zip"
$ZLUDA_SHA256 = "fda8891c6fdfaba438f2eb0f9d749ffa2c1fddbdf225be2301f0d7a25e37208a"
$G1031_URL    = "https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/releases/download/v0.6.2.4/rocm.gfx1031.for.hip.sdk.6.2.4.littlewu.s.logic.7z"
$G1031_SHA256 = "a1946df31e2505b745c284a5234d71efd99ac84dc4530fc20d6aa0656a593b73"
$G1031_DIR    = Join-Path $ROOT "_gfx1031_rocm624\rocm gfx1031 for rocm 6.2.4"   # path rocm_env.py expects

function Write-Header($text) {
    Write-Host ""
    Write-Host ("=" * 64) -ForegroundColor Cyan
    Write-Host "  $text" -ForegroundColor Cyan
    Write-Host ("=" * 64) -ForegroundColor Cyan
}
function Write-Step($n, $total, $text) { Write-Host ""; Write-Host "[$n/$total] $text" -ForegroundColor Yellow }
function Prompt-YesNo($question, $default = $true) {
    if ($Unattended) { return $default }
    $hint = if ($default) { "[Y/n]" } else { "[y/N]" }
    $ans  = Read-Host "$question $hint"
    if ([string]::IsNullOrWhiteSpace($ans)) { return $default }
    return $ans.Trim().ToLower() -in @("y", "yes")
}
function Get-File($url, $dest, $label, $sha256 = $null) {
    Write-Host "  Downloading $label..."
    for ($try = 1; $try -le 3; $try++) {
        try {
            Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
            if ($sha256) {
                $got = (Get-FileHash $dest -Algorithm SHA256).Hash.ToLower()
                if ($got -ne $sha256) { throw "checksum mismatch (got $got)" }
            }
            Write-Host "  ... done." -ForegroundColor Green
            return
        } catch {
            Write-Host "  Attempt $try failed: $_" -ForegroundColor DarkYellow
            Start-Sleep -Seconds 3
        }
    }
    throw "Could not download $label from $url"
}
function Stop-Setup($msg) {
    Write-Host ""
    Write-Host "  ERROR: $msg" -ForegroundColor Red
    if (-not $Unattended) { Read-Host "Press Enter to close" | Out-Null }
    exit 1
}

Write-Header "ImageGen Studio - Setup"
Write-Host "  Folder : $ROOT"
if ($ROOT -match '[!%]') {
    Stop-Setup "The folder path contains '!' or '%', which Windows batch files can't handle. Move the folder (e.g. to C:\ImageGen) and run setup again."
}
New-Item -ItemType Directory -Force $TMPDIR | Out-Null
$TOTAL = 6

# -- 1. Python 3.10 portable --------------------------------------------------
Write-Step 1 $TOTAL "Python 3.10 portable"
if (Test-Path $PYEXE) {
    Write-Host "  Already present: $(& $PYEXE --version 2>&1)" -ForegroundColor Green
} else {
    try {
        $zip = Join-Path $TMPDIR "python-3.10.11-embed-amd64.zip"
        Get-File "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip" $zip "Python 3.10.11 (~8 MB)" `
            "608619f8619075629c9c69f361352a0da6ed7e62f83a0e19c63e0ea32eb7629d"
        New-Item -ItemType Directory -Force $PYTHON | Out-Null
        Expand-Archive -Path $zip -DestinationPath $PYTHON -Force
        # The embeddable build ignores site-packages until "import site" is enabled
        $pth = Join-Path $PYTHON "python310._pth"
        Set-Content $pth "python310.zip`r`n.`r`n`r`nimport site`r`n" -Encoding ASCII
        $getpip = Join-Path $TMPDIR "get-pip.py"
        Get-File "https://bootstrap.pypa.io/get-pip.py" $getpip "pip installer"
        & $PYEXE $getpip --no-warn-script-location
        if ($LASTEXITCODE -ne 0) { throw "get-pip.py failed" }
    } catch { Stop-Setup "Python setup failed: $_" }
}

# -- 2. ZLUDA v6 --------------------------------------------------------------
Write-Step 2 $TOTAL "ZLUDA v6 (runs CUDA programs on AMD GPUs)"
$zludaExe = Join-Path $ROOT "ZLUDA_v6\zluda\zluda.exe"
if (Test-Path $zludaExe) {
    Write-Host "  Already present." -ForegroundColor Green
} else {
    try {
        $zip = Join-Path $TMPDIR "zluda-windows-v6.zip"
        Get-File $ZLUDA_URL $zip "ZLUDA v6 (~34 MB)" $ZLUDA_SHA256
        $dest = Join-Path $ROOT "ZLUDA_v6"
        New-Item -ItemType Directory -Force $dest | Out-Null
        Expand-Archive -Path $zip -DestinationPath $dest -Force        # -> ZLUDA_v6\zluda\
        Set-Content (Join-Path $dest "VERSION.txt") "ZLUDA v6 - zluda-windows-3fe1206.zip from github.com/vosen/ZLUDA/releases/tag/v6" -Encoding ASCII
        if (-not (Test-Path $zludaExe)) { throw "zluda.exe missing after extraction" }
    } catch { Stop-Setup "ZLUDA download failed: $_" }
}

# -- 3. gfx1031 rocBLAS kernels -----------------------------------------------
Write-Step 3 $TOTAL "gfx1031 rocBLAS kernels (RX 6700 / 6700 XT / 6750 XT / 6800M / 6850M XT)"
if (Test-Path (Join-Path $G1031_DIR "library\TensileLibrary_lazy_gfx1031.dat")) {
    Write-Host "  Already present." -ForegroundColor Green
} else {
    try {
        $arc = Join-Path $TMPDIR "rocm.gfx1031.7z"
        Get-File $G1031_URL $arc "gfx1031 kernels (~8 MB)" $G1031_SHA256
        $x = Join-Path $TMPDIR "g1031"
        if (Test-Path $x) { Remove-Item $x -Recurse -Force }
        New-Item -ItemType Directory -Force $x | Out-Null
        # Windows 10 1803+ tar.exe (libarchive) reads .7z
        & "$env:SystemRoot\System32\tar.exe" -xf $arc -C $x
        if ($LASTEXITCODE -ne 0) { throw "could not extract the .7z archive" }
        $lib = Get-ChildItem $x -Recurse -Directory -Filter library | Select-Object -First 1
        if (-not $lib) { throw "no library folder in the archive" }
        New-Item -ItemType Directory -Force $G1031_DIR | Out-Null
        Copy-Item $lib.FullName -Destination $G1031_DIR -Recurse -Force
        $dll = Join-Path $lib.Parent.FullName "rocblas.dll"
        if (Test-Path $dll) { Copy-Item $dll -Destination $G1031_DIR -Force }
    } catch { Stop-Setup "gfx1031 kernel download failed: $_" }
}

# -- 4. Python packages -------------------------------------------------------
Write-Step 4 $TOTAL "Python packages (PyTorch 2.4.1 cu118, diffusers, gradio ...) - ~5 GB, takes a while"
& cmd.exe /d /c "`"$APPDIR\install.bat`" --no-pause"
if ($LASTEXITCODE -ne 0) { Stop-Setup "Package installation failed - see the messages above. Re-run setup to retry." }

# -- 5. AMD HIP SDK -----------------------------------------------------------
Write-Step 5 $TOTAL "AMD HIP SDK 6.x (ZLUDA needs it)"
if ($SkipRocm) {
    Write-Host "  Skipped (-SkipRocm)" -ForegroundColor DarkGray
} else {
    $rocm = $null
    try { $rocm = (& $PYEXE (Join-Path $APPDIR "backend\rocm_env.py") --json) -join "`n" | ConvertFrom-Json } catch { }
    if ($rocm -and $rocm.rocm_bin) {
        Write-Host "  Found: runtime $($rocm.runtime), GPU $($rocm.gpu) ($($rocm.arch))" -ForegroundColor Green
        if ($rocm.note) { Write-Host "  Note: $($rocm.note)" -ForegroundColor DarkYellow }
    } else {
        Write-Host "  Not installed. Install 'AMD HIP SDK for Windows' 6.4 (free, from AMD):" -ForegroundColor DarkYellow
        Write-Host "    https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html" -ForegroundColor Cyan
        Write-Host "  Pick HIP SDK 6.4 (not 7.x). In its installer you can untick the display driver."
        Write-Host "  Then start ImageGen Studio - no need to run this setup again."
        if (-not $Unattended) { Read-Host "  Press Enter to continue" | Out-Null }
    }
}

# -- 6. Models ----------------------------------------------------------------
Write-Step 6 $TOTAL "Starter models (optional, ~6.3 GB)"
$hasModels = (Get-ChildItem (Join-Path $APPDIR "models\checkpoints") -Filter "*.safetensors" -ErrorAction SilentlyContinue).Count -gt 0
if ($SkipModels -or $Unattended) {
    Write-Host "  Skipped. Run ImageGenApp\download_models.bat any time, or use the Civitai tab." -ForegroundColor DarkGray
} elseif ($hasModels) {
    Write-Host "  Checkpoints already present." -ForegroundColor Green
} elseif (Prompt-YesNo "  Download 3 SD 1.5 starter models + VAE (~6.3 GB)?") {
    & $PYEXE (Join-Path $APPDIR "download_models.py")
} else {
    Write-Host "  Skipped. Run ImageGenApp\download_models.bat later, or use the Civitai tab." -ForegroundColor DarkGray
}

Remove-Item $TMPDIR -Recurse -Force -ErrorAction SilentlyContinue
Write-Header "Setup complete"
Write-Host "  1. (Once) check the GPU works:  ImageGenApp\run_zluda.bat selftest_zluda.py"
Write-Host "  2. Start the app:               ImageGenApp\launch.bat"
Write-Host ""
Write-Host "  The FIRST image on a new PC takes 10-15 minutes while ZLUDA compiles GPU kernels."
Write-Host "  It is not stuck. After that, images take seconds."
Write-Host ""
if (-not $Unattended) { Read-Host "Press Enter to close" | Out-Null }
exit 0
