# Utils Tools — Windows PowerShell launcher
# Place this file in any folder on your Windows PATH.
# Handles accented characters (é, à, ü…) in folder names correctly.
#
# If scripts are blocked, run once in an admin PS:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

$OutputEncoding          = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Capture current directory (PowerShell strings are Unicode — no encoding issue)
$winCwd = $PWD.Path

# Convert to WSL path
$wslPath = (& wsl -- wslpath -u $winCwd).Trim()

# Resolve WSL home
$wslHome = (& wsl -- bash -c 'echo $HOME').Trim()

$wslPython = "$wslHome/code/utils/.venv/bin/python3"
$wslScript = "$wslHome/code/utils/utils_tools.py"

# Launch WSL from $env:TEMP (ASCII path) to prevent the relay from trying
# to auto-chdir to the current directory and failing on non-ASCII paths.
Push-Location $env:TEMP
& wsl $wslPython $wslScript --workdir $wslPath
Pop-Location
