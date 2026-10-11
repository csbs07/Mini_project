# PowerShell script to run the LitSearch Full-Text RAG Chatbot
$ErrorActionPreference = "Stop"

Write-Host "Starting Literature Search & Full-Text RAG Chatbot..." -ForegroundColor Cyan

$PythonExe = "python"
if (Test-Path ".\.venv\Scripts\python.exe") {
    $PythonExe = ".\.venv\Scripts\python.exe"
    Write-Host "Using virtual environment: $PythonExe" -ForegroundColor Green
} else {
    Write-Host "Using system python" -ForegroundColor Yellow
}

Write-Host "Navigating to http://127.0.0.1:8000/chat in browser after launch." -ForegroundColor Cyan
& $PythonExe -m uvicorn app.server:app --host 127.0.0.1 --port 8000 --reload
