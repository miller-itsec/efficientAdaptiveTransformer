# Requires: Windows 10/11, PowerShell 7+, Python 3.10/3.11, CUDA (optional)
$ErrorActionPreference = "Stop"

# Robust activation (covers PowerShell variants)
if (Test-Path ".\env\Scripts\activate") {
  & ".\env\Scripts\activate"
} elseif (Test-Path ".\env\Scripts\Activate.ps1") {
  & ".\env\Scripts\Activate.ps1"
} else {
  Write-Host "Virtual environment not found — creating..." -ForegroundColor Yellow
  python -m venv env
  & ".\env\Scripts\activate"
}

Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host "Installing requirements..." -ForegroundColor Cyan
python -m pip install -r .\src\requirements.txt

Write-Host "Setup complete." -ForegroundColor Green
