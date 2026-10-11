# Install the latest weshy release and prepare the machine (Windows x64).
#
#   irm https://raw.githubusercontent.com/kmoshax/weshy/main/scripts/install.ps1 | iex
#
# Environment: WESHY_VERSION (tag, default latest), WESHY_INSTALL_DIR
# (default %LOCALAPPDATA%\Programs\weshy), WESHY_SKIP_SETUP=1 to only install the binary.
$ErrorActionPreference = 'Stop'
$repo = 'kmoshax/weshy'
$target = 'x86_64-pc-windows-msvc'

$tag = $env:WESHY_VERSION
if (-not $tag) { $tag = (Invoke-RestMethod "https://api.github.com/repos/$repo/releases/latest").tag_name }
$dir = $env:WESHY_INSTALL_DIR
if (-not $dir) { $dir = Join-Path $env:LOCALAPPDATA 'Programs\weshy' }
$name = "weshy-$tag-$target"
$base = "https://github.com/$repo/releases/download/$tag"

$tmp = Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    Write-Host "downloading weshy $tag for $target"
    Invoke-WebRequest "$base/$name.zip" -OutFile "$tmp\$name.zip"
    Invoke-WebRequest "$base/SHA256SUMS" -OutFile "$tmp\SHA256SUMS"
    $expected = (Select-String -Path "$tmp\SHA256SUMS" -Pattern " $name.zip$").Line.Split(' ')[0]
    $actual = (Get-FileHash "$tmp\$name.zip" -Algorithm SHA256).Hash.ToLower()
    if ($expected -ne $actual) { throw "checksum mismatch for $name.zip" }
    Expand-Archive "$tmp\$name.zip" -DestinationPath $tmp
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Copy-Item "$tmp\$name\*" $dir -Force
} finally {
    Remove-Item -Recurse -Force $tmp
}
Write-Host "installed $dir\weshy.exe"

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if (($userPath -split ';') -notcontains $dir) {
    [Environment]::SetEnvironmentVariable('Path', "$userPath;$dir", 'User')
    Write-Host "added $dir to your user PATH (open a new terminal to use it)"
}

if ($env:WESHY_SKIP_SETUP -ne '1') { & "$dir\weshy.exe" setup @args }
