# Utils Tools — Windows PowerShell launcher
# Place this file in any folder on your Windows PATH.
# Handles accented characters (é, à, ü…) in folder names correctly.
#
# If scripts are blocked, run once in an admin PS:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

$OutputEncoding          = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Capture current directory.
# Use ProviderPath, NOT .Path — when the cwd is a UNC path into WSL
# (\\wsl.localhost\Ubuntu\…), .Path carries a provider prefix
# ("Microsoft.PowerShell.Core\FileSystem::\\wsl.localhost\…") that breaks
# wslpath. ProviderPath gives the bare filesystem path.
$winCwd = $PWD.ProviderPath

# Default distro for the wsl calls below; overridden when cwd lives inside a
# specific distro's UNC path so $HOME / python resolve in the same distro.
$distroArgs = @()

# If we're already on a WSL path (\\wsl.localhost\<distro>\… or \\wsl$\<distro>\…),
# convert it directly to a Linux path — wslpath mishandles UNC-into-WSL and the
# cross-distro case. Otherwise fall back to wslpath for genuine Windows paths
# (C:\…  →  /mnt/c/…).
$wslUnc = [regex]::Match($winCwd, '^\\\\wsl(?:\.localhost|\$)\\([^\\]+)\\(.*)$')
if ($wslUnc.Success) {
    $distro     = $wslUnc.Groups[1].Value
    $rest       = $wslUnc.Groups[2].Value
    $distroArgs = @('-d', $distro)
    $wslPath    = '/' + ($rest -replace '\\', '/')
} else {
    $wslPath = (& wsl @distroArgs -- wslpath -u $winCwd).Trim()
}

# Resolve WSL home (in the same distro as the cwd)
$wslHome = (& wsl @distroArgs -- bash -c 'echo $HOME').Trim()

$wslPython = "$wslHome/code/utils/.venv/bin/python3"
$wslScript = "$wslHome/code/utils/utils_tools.py"

# Launch WSL from $env:TEMP (ASCII path) to prevent the relay from trying
# to auto-chdir to the current directory and failing on non-ASCII paths.
Push-Location $env:TEMP
& wsl @distroArgs $wslPython $wslScript --workdir $wslPath
Pop-Location
