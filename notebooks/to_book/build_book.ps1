# Build the ETE 2526 Quarto book
# Notebooks are rendered using their stored outputs — no kernel re-execution.
#
# Usage:
#   .\build_book.ps1          # render to _book/
#   .\build_book.ps1 -Preview # render + open live-reload preview server

param(
  [switch]$Preview
)

Write-Host "Building ETE 2526 book..." -ForegroundColor Cyan

if ($Preview) {
  quarto preview
} else {
  quarto render
  if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Done. Open _book/index.html in a browser." -ForegroundColor Green
    Write-Host "Or run:  .\build_book.ps1 -Preview  for a live-reload server." -ForegroundColor DarkGray
  } else {
    Write-Host "Build failed — see errors above." -ForegroundColor Red
    Write-Host "If the error mentions a missing kernel, the notebooks are" -ForegroundColor Yellow
    Write-Host "trying to re-execute. Set 'freeze: true' in _quarto.yml." -ForegroundColor Yellow
  }
}
