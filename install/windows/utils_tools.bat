@echo off
:: Utils Tools — Windows launcher
:: Place this file in any folder that is on your Windows PATH.
:: It will forward the current Windows directory to the WSL Python TUI.
::
:: NOTE: prefer utils_tools.ps1 if you use PowerShell — it handles
::       accented characters (é, à, ü…) and WSL UNC paths more reliably.

:: Switch console to UTF-8 so wslpath output is not mangled by CP1252
chcp 65001 >nul 2>&1

setlocal enabledelayedexpansion

:: Capture current directory before changing it
set "WIN_CWD=%CD%"
set "DISTRO_ARGS="

:: If the cwd is already a WSL UNC path (\\wsl.localhost\<distro>\… or
:: \\wsl$\<distro>\…), convert it directly — wslpath mishandles UNC-into-WSL.
:: Otherwise fall back to wslpath for genuine Windows paths (C:\… → /mnt/c/…).
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
    for /f "delims=" %%P in ('wsl -- wslpath -u "%WIN_CWD%"') do set "WSL_CWD=%%P"
)

:: Resolve WSL home directory (in the same distro as the cwd)
for /f "delims=" %%H in ('wsl %DISTRO_ARGS% -- bash -c "echo $HOME"') do set "WSL_HOME=%%H"

set "WSL_PYTHON=%WSL_HOME%/code/utils/.venv/bin/python3"
set "WSL_SCRIPT=%WSL_HOME%/code/utils/utils_tools.py"

:: Launch WSL from %TEMP% (ASCII path) to prevent the relay from trying
:: to auto-chdir to the current directory and failing on non-ASCII paths.
pushd %TEMP%
wsl %DISTRO_ARGS% "%WSL_PYTHON%" "%WSL_SCRIPT%" --workdir "%WSL_CWD%"
popd

endlocal
