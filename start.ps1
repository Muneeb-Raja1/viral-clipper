# Run from the repo root: .\start.ps1
Set-Location $PSScriptRoot
Write-Host "Starting Viral Clipper..." -ForegroundColor Cyan
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
