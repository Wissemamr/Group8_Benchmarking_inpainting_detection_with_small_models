# Incremental backup of the AEROBLADE cache (reconstructions, LPIPS maps, per-image records, run
# logs, cached localization scores): the part of the results that costs ~17 s per image to recompute.
#
#   powershell -ExecutionPolicy Bypass -File scripts\backup_cache.ps1 -Dest C:\backups\aeroblade
#   ... -EveryMinutes 30                       repeat until Ctrl+C
#   ... -EveryMinutes 30 -StopWhenMissing F    repeat until file F disappears, then back up once more
#
# Only adds / updates files, never deletes from the backup (no /MIR), so a deleted or corrupted
# cache cannot propagate. Restore with:  robocopy <Dest> <AEROBLADE>\cache /E
param(
    [Parameter(Mandatory = $true)][string]$Dest,
    [int]$EveryMinutes = 0,
    [string]$StopWhenMissing = ""
)

$src = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..")).Path "cache"

function Backup-Once {
    robocopy $src $Dest /E /R:1 /W:1 /NFL /NDL /NJH /NP /XF "*.tmp.*" ".run_in_progress" | Out-Null
    if ($LASTEXITCODE -ge 8) { Write-Warning "robocopy failed (exit code $LASTEXITCODE)" }
    else { Write-Output "$(Get-Date -Format s) backup ok -> $Dest" }
}

New-Item -ItemType Directory -Force $Dest | Out-Null
Backup-Once
if ($EveryMinutes -gt 0) {
    while ($true) {
        Start-Sleep -Seconds (60 * $EveryMinutes)
        $running = (-not $StopWhenMissing) -or (Test-Path $StopWhenMissing)
        Backup-Once
        if (-not $running) { break }
    }
}
