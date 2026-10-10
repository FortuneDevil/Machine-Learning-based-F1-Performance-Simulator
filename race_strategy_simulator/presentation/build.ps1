# Rebuild the presentation: numbers and figures, then .pptx, then .pdf.
# Usage:  powershell -File presentation\build.ps1 [-SkipAnalysis]
param([switch]$SkipAnalysis)

$project = Split-Path -Parent $PSScriptRoot
$python = "$env:USERPROFILE\.venvs\f1sim\Scripts\python.exe"
$soffice = "C:\Program Files\LibreOffice\program\soffice.exe"
$deck = Join-Path $PSScriptRoot "F1_Race_Strategy_Simulator.pptx"

Push-Location $project
if (-not $SkipAnalysis) { & $python presentation\deck_analysis.py | Out-Null }
Remove-Item (Join-Path $PSScriptRoot "assets\equations") -Recurse -Force -ErrorAction SilentlyContinue
& $python presentation\build_deck.py $project $deck
& $soffice --headless --convert-to pdf --outdir $PSScriptRoot $deck | Out-Null
Pop-Location
