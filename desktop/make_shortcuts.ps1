<#
  Creates "ImageGen Studio" shortcuts on the Desktop and in the Start menu that open the
  desktop window. Uses the packaged exe (dist\...\ImageGen Studio.exe) when it exists,
  otherwise runs the window straight from this folder with node_modules\electron.
  Usage:  powershell -ExecutionPolicy Bypass -File desktop\make_shortcuts.ps1
#>
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$exe  = Join-Path $here "dist\ImageGen Studio-win32-x64\ImageGen Studio.exe"
$icon = Join-Path $here "assets\icon.ico"
if (Test-Path $exe) {
    $target = $exe; $lnkArgs = ""; $workdir = Split-Path $exe
} else {
    $target = Join-Path $here "node_modules\electron\dist\electron.exe"
    if (-not (Test-Path $target)) { throw "Run 'npm install' in $here first (or build with 'npm run package')." }
    $lnkArgs = "`"$here`""; $workdir = $here
}
$shell = New-Object -ComObject WScript.Shell
$places = @(
    [Environment]::GetFolderPath("Desktop"),
    (Join-Path ([Environment]::GetFolderPath("Programs")) "")
)
foreach ($dir in $places) {
    $lnk = $shell.CreateShortcut((Join-Path $dir "ImageGen Studio.lnk"))
    $lnk.TargetPath = $target
    $lnk.Arguments = $lnkArgs
    $lnk.WorkingDirectory = $workdir
    $lnk.IconLocation = "$icon,0"
    $lnk.Description = "ImageGen Studio - local Stable Diffusion on your AMD GPU"
    $lnk.Save()
    Write-Host "Created $(Join-Path $dir 'ImageGen Studio.lnk')"
}
