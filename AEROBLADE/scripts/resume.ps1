# Start or resume the AEROBLADE run from Windows PowerShell (it runs inside WSL).
#
#   powershell -ExecutionPolicy Bypass -File scripts\resume.ps1
#
# Safe to run any number of times: finished images are skipped (see README, "Resume and backup").
# Keeps Windows awake while it runs (no power setting is changed: the request ends with this
# process) and appends the output to cache\run_all.log. Extra arguments go to run_aeroblade.py,
# e.g.  -ExtraArgs "--threads 4".
# Assumes the setup used for this project: WSL with conda at ~/miniforge3 and an env "aeroblade".
param([string]$ExtraArgs = "")

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$wslRoot = (wsl -e wslpath -a "$root").Trim()   # -e: no shell, so the backslashes survive
New-Item -ItemType Directory -Force (Join-Path $root "cache") | Out-Null

Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
# Decimal, not 0x80000001: Windows PowerShell 5.1 reads that literal as a negative Int32, which
# cannot convert to the UInt32 parameter.
[void][W.P]::SetThreadExecutionState([uint32]2147483649)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
try {
    wsl -e bash -c "source ~/miniforge3/etc/profile.d/conda.sh; conda activate aeroblade; cd '$wslRoot'; export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1; bash scripts/run_all.sh $ExtraArgs 2>&1 | tee -a cache/run_all.log"
} finally {
    [void][W.P]::SetThreadExecutionState([uint32]2147483648)   # ES_CONTINUOUS only: release
}
