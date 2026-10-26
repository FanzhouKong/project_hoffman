@echo off
rem Windows: double-click to start the peak3d app in your browser (needs Docker Desktop).
rem The app sees your user folder (C:\Users\you) as /data. Optional: set PEAK3D_DATA or PEAK3D_IMAGE first.
setlocal EnableExtensions
title peak3d
if "%PEAK3D_IMAGE%"=="" set "PEAK3D_IMAGE=ghcr.io/fanzhoukong/peak3d:gui-prototype"
if "%PEAK3D_DATA%"=="" set "PEAK3D_DATA=%USERPROFILE%"
set "NAME=peak3d-gui"
echo peak3d
echo ------

rem 1. Docker installed and running
where docker >nul 2>nul
if errorlevel 1 (
  echo Docker is not installed. Install Docker Desktop from the page that just opened,
  echo start it once, then double-click this file again.
  start "" "https://www.docker.com/products/docker-desktop/"
  pause
  exit /b 1
)
docker info >nul 2>nul
if not errorlevel 1 goto docker_ok
echo Starting Docker Desktop...
if exist "%ProgramFiles%\Docker\Docker\Docker Desktop.exe" start "" "%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
for /l %%i in (1,1,90) do (
  docker info >nul 2>nul && goto docker_ok
  timeout /t 2 /nobreak >nul
)
echo Docker is not running. Start Docker Desktop, wait until it says it is running, and try again.
pause
exit /b 1
:docker_ok

rem 2. Already running? Just open it again (does not interrupt a run in progress)
set "RUNNING="
for /f %%r in ('docker inspect -f "{{.State.Running}}" %NAME% 2^>nul') do set "RUNNING=%%r"
if /i not "%RUNNING%"=="true" goto start_new
for /f "tokens=2 delims==" %%t in ('docker inspect -f "{{range .Config.Env}}{{println .}}{{end}}" %NAME% ^| findstr /b "PEAK3D_TOKEN="') do set "TOKEN=%%t"
call :find_url
start "" "%URL%"
echo peak3d is already running: %URL%
timeout /t 5 >nul
exit /b 0
:start_new

rem 3. Latest image (about 300 MB the first time; later only what changed)
echo Checking for the latest peak3d (the first time downloads about 300 MB)...
docker pull "%PEAK3D_IMAGE%"
if errorlevel 1 (
  docker image inspect "%PEAK3D_IMAGE%" >nul 2>nul
  if errorlevel 1 (
    echo Could not download peak3d. Check your internet connection and try again.
    pause
    exit /b 1
  )
  echo Could not check for updates; using the copy on this computer.
)

rem 4. Start: your folder as /data, a random free port on this computer only, a fresh token
for /f %%t in ('powershell -NoProfile -Command "[guid]::NewGuid().ToString('N')"') do set "TOKEN=%%t"
docker rm -f %NAME% >nul 2>nul
docker run -d --rm --init --name %NAME% -p 127.0.0.1::8765 -e PEAK3D_TOKEN=%TOKEN% -v peak3d-cache:/var/cache/peak3d -v "%PEAK3D_DATA%:/data" "%PEAK3D_IMAGE%" gui --host 0.0.0.0 --port 8765 --root /data --no-browser >nul
if errorlevel 1 (
  echo Could not start peak3d ^(see the message above^).
  pause
  exit /b 1
)
call :find_url
for /l %%i in (1,1,60) do (
  powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Headers @{'X-Peak3d-Token'='%TOKEN%'} 'http://127.0.0.1:%PORT%/api/info' | Out-Null; exit 0 } catch { exit 1 }" && goto ready
  timeout /t 1 /nobreak >nul
)
:ready
start "" "%URL%"
echo.
echo peak3d is running at:
echo     %URL%
echo Your folder %PEAK3D_DATA% appears in peak3d as /data.
echo.
echo Keep this window open while you use peak3d. Press any key here to stop peak3d.
pause >nul
docker stop %NAME% >nul
exit /b 0

:find_url
set "PORT="
for /f "tokens=2 delims=:" %%p in ('docker port %NAME% 8765/tcp') do if not defined PORT set "PORT=%%p"
set "URL=http://127.0.0.1:%PORT%/?token=%TOKEN%"
exit /b 0
