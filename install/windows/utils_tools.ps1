# Utils Tools - Windows PowerShell launcher
# Place this file in any folder on your Windows PATH.
# Handles accented characters (e, a, u...) in folder names correctly.
#
# If scripts are blocked, run once in an admin PS:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#
# Kept pure ASCII on purpose: Windows PowerShell 5.1 reads BOM-less files as
# ANSI, so non-ASCII bytes here would be re-interpreted. The paths it handles
# at RUNTIME can of course contain any character.
#
# Normally installed by:  bash install/windows/install_launchers.sh  (from WSL),
# which fills in $UtilsDistro / $UtilsDir below with the real values. Copying
# this file by hand also works: empty values fall back to auto-detection
# (default distro, repo at ~/code/utils).

$OutputEncoding          = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# -- Configuration (filled in by install_launchers.sh) ------------------------
$UtilsDistro = ''
$UtilsDir    = ''

# Capture current directory.
# Use ProviderPath, NOT .Path - when the cwd is a UNC path into WSL
# (\\wsl.localhost\Ubuntu\...), .Path carries a provider prefix
# ("Microsoft.PowerShell.Core\FileSystem::\\wsl.localhost\...") that breaks
# wslpath. ProviderPath gives the bare filesystem path.
$winCwd = $PWD.ProviderPath

# Default distro for the wsl calls below; overridden when cwd lives inside a
# specific distro's UNC path so $HOME / python resolve in the same distro.
$distroArgs = @()
if ($UtilsDistro) { $distroArgs = @('-d', $UtilsDistro) }

# If we're already on a WSL path (\\wsl.localhost\<distro>\... or \\wsl$\<distro>\...),
# convert it directly to a Linux path - wslpath mishandles UNC-into-WSL and the
# cross-distro case. Otherwise fall back to wslpath for genuine Windows paths
# (C:\...  ->  /mnt/c/...).
$wslUnc = [regex]::Match($winCwd, '^\\\\wsl(?:\.localhost|\$)\\([^\\]+)\\(.*)$')
if ($wslUnc.Success) {
    $distro     = $wslUnc.Groups[1].Value
    $rest       = $wslUnc.Groups[2].Value
    $distroArgs = @('-d', $distro)
    $wslPath    = '/' + ($rest -replace '\\', '/')
} elseif ($winCwd -match '^([A-Za-z]):\\(.*)$') {
    # Genuine Windows drive path (C:\Users\...). Convert to /mnt/<drive>/... here:
    # passing backslashes through `wsl -- wslpath` strips them
    # (C:\Users\X -> C:UsersX), so wslpath fails and returns null -> .Trim() throws.
    # Map it directly instead.
    $drive   = $Matches[1].ToLower()
    $rest    = $Matches[2] -replace '\\', '/'
    $wslPath = "/mnt/$drive/$rest"
} else {
    # Last resort for anything exotic (mapped network drive, etc.).
    $wslPath = (& wsl @distroArgs -- wslpath -u $winCwd).Trim()
}

# Resolve the utils checkout. Baked in at install time; otherwise assume the
# documented default (~/code/utils) inside the distro we are talking to.
if (-not $UtilsDir) {
    $wslHome  = (& wsl @distroArgs -- bash -c 'echo $HOME').Trim()
    $UtilsDir = "$wslHome/code/utils"
}

$wslPython = "$UtilsDir/.venv/bin/python3"
$wslScript = "$UtilsDir/utils_tools.py"

# Launch WSL from $env:TEMP (ASCII path) to prevent the relay from trying
# to auto-chdir to the current directory and failing on non-ASCII paths.
Push-Location $env:TEMP
& wsl @distroArgs $wslPython $wslScript --workdir $wslPath
Pop-Location
