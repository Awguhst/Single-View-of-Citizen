<#
Activates this project's local conda environment (.\env) and starts the
FastAPI dev server (uvicorn, with auto-reload) in the same step.

Usage (from the project root, in PowerShell):
    .\run.ps1
    .\run.ps1 -Port 8001
#>

param(
    [int]$Port = 8700
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

$CondaExe = (Get-Command conda -ErrorAction SilentlyContinue).Source
if (-not $CondaExe) {
    $CondaExe = "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
}
if (-not (Test-Path $CondaExe)) {
    throw "Could not find conda.exe. Install Miniconda/Anaconda, or edit `$CondaExe at the top of run.ps1 to point at your install."
}

# Load the conda shell hook so `conda activate` works even when this script
# runs without the interactive profile that normally wires it up.
(& $CondaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate "$ProjectRoot\env"

Set-Location $ProjectRoot
uvicorn app.main:app --reload --port $Port
