# Internal helper. A JSON file keeps paths/arguments out of executable PowerShell source.
param([Parameter(Mandatory=$true)][string]$ConfigPath)
$ErrorActionPreference = 'Stop'
$config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$previous = Get-Process -Id $config.pid -ErrorAction SilentlyContinue
if ($previous) { $previous.WaitForExit() }
$target = [string]$config.target
$exe = Join-Path $target 'wessh.exe'
$backup = Join-Path (Split-Path -Parent $ConfigPath) 'previous'
$changed = @()
try {
    if ($config.installer) {
        $arguments = @('/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/NOCLOSEAPPLICATIONS', '/NORESTARTAPPLICATIONS', ('/DIR="' + $target + '"'))
        $process = Start-Process -FilePath $config.source -ArgumentList $arguments -Wait -PassThru
        if ($process.ExitCode -ne 0) { throw "Installer exited with code $($process.ExitCode)" }
    } else {
        New-Item -ItemType Directory -Path $backup | Out-Null
        foreach ($name in @('wessh.exe', 'wessh-cli.exe')) {
            $destination = Join-Path $target $name
            if (Test-Path -LiteralPath $destination) { Move-Item -LiteralPath $destination -Destination (Join-Path $backup $name) }
            $changed += $name
            Copy-Item -LiteralPath (Join-Path $config.source $name) -Destination $destination
        }
    }
    Write-Output 'Update installed.'
} catch {
    Write-Output "Update failed: $_"
    foreach ($name in $changed) {
        $destination = Join-Path $target $name
        Remove-Item -LiteralPath $destination -Force -ErrorAction SilentlyContinue
        $saved = Join-Path $backup $name
        if (Test-Path -LiteralPath $saved) { Move-Item -LiteralPath $saved -Destination $destination }
    }
} finally {
    if ($config.restart -and (Test-Path -LiteralPath $exe)) {
        $arguments = @($config.arguments | ForEach-Object { '"' + ($_ -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"' })
        if ($arguments.Count -gt 0) { Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $target }
        else { Start-Process -FilePath $exe -WorkingDirectory $target }
    }
}
