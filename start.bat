@echo off
title AI Vision Studio Server
echo ==========================================
echo Starting AI Vision Studio Local Server...
echo ==========================================
echo.
echo Press CTRL+C to stop the server at any time.
echo.

:: Go into the website folder
cd public

:: Open the browser automatically
start http://localhost:8000

:: Start the Python server
python -m http.server 8000

pause