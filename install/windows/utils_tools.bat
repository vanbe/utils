@echo off
:: Utils Tools - Windows launcher (CMD)
:: Place this file in any folder that is on your Windows PATH.
:: It forwards the current Windows directory to the WSL Python TUI.
::
:: NOTE: prefer utils_tools.ps1 if you use PowerShell - it handles accented
::       characters and WSL UNC paths more reliably.
::
:: WARNING - THIS FILE MUST STAY PURE ASCII. cmd.exe re-reads a .bat from a
:: saved BYTE offset after every command; a multi-byte UTF-8 character shifts
:: that offset and the interpreter resumes mid-line, executing garbage
:: ("'ISTRO' is not recognized as an internal or external command"). Accented
:: comments are enough to break the whole launcher. Keep it 7-bit.
::
:: Normally installed by:  bash install/windows/install_launchers.sh  (from WSL),
:: which fills in UTILS_DISTRO / UTILS_DIR below with the real values. Copying
:: this file by hand also works: empty values fall back to auto-detection
:: (default distro, repo at ~/code/utils).

:: Switch console to UTF-8 so wslpath output is not mangled by CP1252
chcp 65001 >nul 2>&1

setlocal enabledelayedexpansion

:: -- Configuration (filled in by install_launchers.sh) -----------------------
set "UTILS_DISTRO="
set "UTILS_DIR="

:: Capture current directory before changing it
set "WIN_CWD=%CD%"
set "DISTRO_ARGS="

:: If the cwd is already a WSL UNC path (\\wsl.localhost\<distro>\... or
:: \\wsl$\<distro>\...), convert it directly - wslpath mishandles UNC-into-WSL.
:: Otherwise fall back to wslpath for genuine Windows paths (C:\... -> /mnt/c/...).
set "WSL_CWD="
echo %WIN_CWD% | findstr /i /r "^\\\\wsl.localhost\\ ^\\\\wsl\$\\" >nul
if not errorlevel 1 (
    for /f "tokens=4,* delims=\" %%A in ("%WIN_CWD%") do (
        set "DISTRO=%%A"
        set "REST=%%B"
    )
    set "DISTRO_ARGS=-d !DISTRO!"
    set "REST=/!REST:\=/!"
    set "WSL_CWD=!REST!"
) else (
    if defined UTILS_DISTRO set "DISTRO_ARGS=-d !UTILS_DISTRO!"
    for /f "delims=" %%P in ('wsl !DISTRO_ARGS! -- wslpath -u "%WIN_CWD%"') do set "WSL_CWD=%%P"
)

:: Resolve the utils checkout. Baked in at install time; otherwise assume the
:: documented default (~/code/utils) inside the distro we are talking to.
if not defined UTILS_DIR (
    for /f "delims=" %%H in ('wsl !DISTRO_ARGS! -- bash -c "echo $HOME"') do set "WSL_HOME=%%H"
    set "UTILS_DIR=!WSL_HOME!/code/utils"
)

set "WSL_PYTHON=!UTILS_DIR!/.venv/bin/python3"
set "WSL_SCRIPT=!UTILS_DIR!/utils_tools.py"

:: Launch WSL from %TEMP% (ASCII path) to prevent the relay from trying
:: to auto-chdir to the current directory and failing on non-ASCII paths.
pushd %TEMP%
wsl !DISTRO_ARGS! "!WSL_PYTHON!" "!WSL_SCRIPT!" --workdir "!WSL_CWD!"
popd

endlocal
