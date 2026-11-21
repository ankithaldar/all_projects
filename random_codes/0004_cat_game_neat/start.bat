@echo off
setlocal
rem Cat Game NEAT - local static-server launcher (Windows)
rem Serves ONLY this project directory, bound EXPLICITLY to 127.0.0.1,
rem on an ephemeral port generated locally. Direct file:// opening is
rem unsupported by browsers, so use this launcher to play the game.

rem (a) Serve only the script's own directory (the project root).
cd /d "%~dp0"

rem Pick a Python interpreter: prefer the "py -3" launcher, fall back to python.
set "PY=python"
where py >nul 2>&1 && (py -3 --version >nul 2>&1 && set "PY=py -3")
%PY% --version >nul 2>&1 || goto :nopython

rem (c) Locally generate a validated integer port via socket bind-to-port-0.
rem     The command is a fixed literal; no env/CLI strings are evaluated.
set "PORT="
for /f "usebackq delims=" %%P in (`%PY% -c "import socket;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.bind(('127.0.0.1',0));p=s.getsockname()[1];s.close();assert isinstance(p,int) and 1<=p<=65535;print(p)"`) do set "PORT=%%P"
if not defined PORT goto :badport
echo %PORT%| findstr /r /c:"^[0-9][0-9]*$" >nul || goto :badport

rem (b)/(c) The port is used only as a discrete argument and in this fixed URL.
set "URL=http://127.0.0.1:%PORT%/index.html"
echo Starting local static server at %URL%

rem (b) Bind loopback EXPLICITLY. Start in the background so checks can run.
start "CatGameStaticServer" /b %PY% -m http.server %PORT% --bind 127.0.0.1 --directory .

rem (d) Wait for the server to answer on loopback.
set /a tries=0
:readyloop
%PY% -c "import socket,sys;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.settimeout(1);s.connect(('127.0.0.1',int(sys.argv[1])));s.close()" %PORT% >nul 2>&1
if not errorlevel 1 goto :readyok
set /a tries+=1
if %tries% GEQ 10 goto :readyfail
timeout /t 1 /nobreak >nul
goto :readyloop
:readyok
echo CHECK OK: reachable on loopback 127.0.0.1:%PORT%
goto :afterready
:readyfail
echo CHECK FAIL: server not reachable on loopback >&2
goto :fail
:afterready

rem (d) Confirm NOT reachable on a local non-loopback interface.
set "NBIP="
for /f "usebackq delims=" %%I in (`%PY% -c "import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.connect(('192.0.2.1',9));print(s.getsockname()[0])"`) do set "NBIP=%%I"
if not defined NBIP (
  echo CHECK WARN: no non-loopback interface to probe; --bind 127.0.0.1 still enforced
  goto :checkspass
)
%PY% -c "import socket,sys;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.settimeout(2);s.connect((sys.argv[1],int(sys.argv[2])));s.close()" %NBIP% %PORT% >nul 2>&1
if not errorlevel 1 (
  echo CHECK FAIL: server IS reachable on non-loopback %NBIP%:%PORT%
  goto :fail
)
echo CHECK OK: NOT reachable on non-loopback %NBIP%:%PORT% (loopback-only)
:checkspass

rem Open the fixed URL in the default browser (best effort).
start "" "%URL%"

echo.
echo ============================================================
echo  Cat Game NEAT is served at: %URL%
echo  Opening it in your browser now.
echo  This server listens on 127.0.0.1 ONLY - not on your network.
echo  Press Ctrl-C to stop the server.
echo ============================================================
echo.

rem Keep the batch (and its background server) alive until Ctrl-C.
:waitloop
ping -n 2 127.0.0.1 >nul
goto :waitloop

:nopython
echo python3 is required but was not found. >&2
exit /b 1

:badport
echo Failed to generate a valid local port. >&2
exit /b 1

:fail
echo Security/startup checks FAILED. Stopping server. >&2
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*http.server*%PORT%*' } | Stop-Process -Force" >nul 2>&1
exit /b 1
