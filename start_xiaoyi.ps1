# ============================================================
#  Xiaoyi (Reachy Mini) launcher for Windows PowerShell.
#  Ported from start_mac.sh, adapted to Windows realities.
#
#  Usage:
#     .\start_xiaoyi.ps1                 # normal: wake-word on, runs until Ctrl+C
#     .\start_xiaoyi.ps1 -NoWake         # skip wake word (connect immediately; quick smoke test)
#     .\start_xiaoyi.ps1 -Seconds 120    # clean auto-exit after N seconds (no Ctrl+C needed)
#     .\start_xiaoyi.ps1 -Restart        # force-restart the robot daemon first
#     .\start_xiaoyi.ps1 -FaceMp         # use MediaPipe face backend (default = SCRFD/YuNet)
#     .\start_xiaoyi.ps1 stop            # stop main + daemon (robot sleeps)
#
#  Exit: press Ctrl+C in THIS window -> graceful shutdown
#        (re-center head/body, flush memory + gallery, restore auto body-yaw).
#        The 'stop' subcommand is a HARD kill (no graceful flush) -- use only to
#        clear a wedged run; prefer Ctrl+C in the run window.
#  Dashboard while running: http://localhost:7654
#
#  NOTE (Windows adaptations vs start_mac.sh):
#   - daemon comes up via tools\daemon_up.py (power/overload exit codes), not a
#     raw reachy-mini-daemon + control_mode poll.
#   - d01 runs in the FOREGROUND (not nohup background): Windows has no clean
#     SIGTERM, so background + Stop-Process would hard-kill and drop in-session
#     memory. Foreground Ctrl+C gives the graceful finally (flush + re-center).
# ============================================================
[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$Command = "",  # 'stop' -> stop main + daemon
    [switch]$NoWake,       # add --no-wake
    [int]$Seconds = 0,     # >0 -> pass run-seconds to d01 for a clean timed exit; 0 = run forever
    [switch]$Restart,      # pass --restart to daemon_up
    [switch]$FaceMp        # FACE_BACKEND=mediapipe (mirror start_mac.sh --face-mp)
)

Set-Location -LiteralPath $PSScriptRoot

# ---- force UTF-8 so Chinese/emoji from python (PYTHONUTF8=1) don't mojibake ----
# Chinese Windows consoles default to GBK (cp936); python emits UTF-8, and piping
# through Tee-Object makes PowerShell decode it with [Console]::OutputEncoding.
# If that's GBK, UTF-8 bytes are misread -> garbled Chinese. Pin everything to UTF-8.
try {
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    [Console]::InputEncoding  = [System.Text.UTF8Encoding]::new($false)
    chcp 65001 > $null 2>&1
} catch { }

# ---- paths ----
$py        = "C:\Users\ldkji\AppData\Local\Reachy Mini Control\.venv\Scripts\python.exe"
$daemonExe = "reachy-mini-daemon.exe"
$logDir    = Join-Path $PSScriptRoot "log"
$mainLog   = Join-Path $logDir "main.log"
$visPort   = 7654
$url       = "http://localhost:$visPort"

function Info($m) { Write-Host "[INFO]  $m" -ForegroundColor Green }
function Warn($m) { Write-Host "[WARN]  $m" -ForegroundColor Yellow }
function Fail($m) { Write-Host "[ERROR] $m" -ForegroundColor Red; exit 1 }

# ---- stop subcommand (mirror start_mac.sh: bash start_mac.sh stop) ----
function Stop-All {
    Info "Stopping Xiaoyi main (d01)..."
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -match 'd01_realtime_chat' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 1
    Info "Stopping reachy daemon (robot will sleep)..."
    taskkill /F /IM $daemonExe 2>$null | Out-Null
    Info "Done. (hard kill: no graceful flush -- prefer Ctrl+C in the run window)"
    exit 0
}

if ($Command -eq "stop") { Stop-All }
if ($Command -ne "")     { Fail "Unknown command '$Command' (only 'stop' is supported)." }

# ---- 1. check venv python ----
if (-not (Test-Path -LiteralPath $py)) {
    Write-Host "[ERROR] Reachy venv python not found:" -ForegroundColor Red
    Write-Host "        $py" -ForegroundColor Red
    Write-Host "        Edit `$py at the top of this script if your install path differs." -ForegroundColor Yellow
    exit 1
}

# ---- 2. environment (mirror start_mac.sh) ----
# Do NOT set REALTIME_MODEL -> defaults to qwen3.5-omni-plus-realtime.
# (Memory/tool-calling REQUIRE the plus model; do not use flash.)
$env:PYTHONUTF8       = "1"
$env:PYTHONUNBUFFERED = "1"
$env:VIS_DEBUG        = "1"
$env:HF_HUB_OFFLINE   = "1"                       # no HuggingFace network (models are cached)
$env:NO_PROXY         = "localhost,127.0.0.1,::1"
$env:no_proxy         = "localhost,127.0.0.1,::1"
if ($FaceMp) {
    $env:FACE_BACKEND = "mediapipe"
    Info "Face backend: MediaPipe (-FaceMp)"
} else {
    Info "Face backend: default SCRFD/YuNet (use -FaceMp to switch to MediaPipe)"
}

Write-Host ""
Write-Host "  ==============================================" -ForegroundColor Cyan
Write-Host "     Xiaoyi (Reachy Mini) launcher -- Windows"    -ForegroundColor Cyan
Write-Host "  ==============================================" -ForegroundColor Cyan
Write-Host ""

# ---- 3. free VIS_DEBUG port if a stray d01 still holds it (avoids 'address in use') ----
try {
    $held = Get-NetTCPConnection -LocalPort $visPort -State Listen -ErrorAction Stop
    foreach ($c in $held) {
        Warn "Port $visPort held by PID $($c.OwningProcess); killing stray..."
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 500
} catch { }

# ---- 4. power on the robot (daemon_up: power/overload exit codes) ----
Info "1/2 Powering on robot (daemon_up)..."
$daemonArgs = @("tools\daemon_up.py")
if ($Restart) { $daemonArgs += "--restart" }
& $py @daemonArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Power-on FAILED (exit=$LASTEXITCODE)." -ForegroundColor Red
    Write-Host "        exit 2 = check power adapter/switch (USB present != motors powered)." -ForegroundColor Yellow
    Write-Host "        exit 3 = power-cycle to clear motor overload, then retry." -ForegroundColor Yellow
    exit $LASTEXITCODE
}
Info "Daemon ready (control_mode enabled)."

# ---- 5. log dir; main.log is overwritten each run (like start_mac.sh) ----
if (-not (Test-Path -LiteralPath $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

# ---- 6. VIS_DEBUG: open the Dashboard once its port answers (background waiter) ----
$openJob = Start-Job -ScriptBlock {
    param($u)
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 1
        try {
            $r = Invoke-WebRequest -Uri $u -TimeoutSec 1 -UseBasicParsing -ErrorAction Stop
            if ($r.StatusCode -eq 200) { Start-Process $u; return }
        } catch { }
    }
} -ArgumentList $url

# ---- 7. launch Xiaoyi (FOREGROUND; Ctrl+C = graceful exit; output -> UTF-8 main.log) ----
$d01Args = @("-u", "voice\d01_realtime_chat.py")
if ($NoWake)        { $d01Args += "--no-wake" }
if ($Seconds -gt 0) { $d01Args += "$Seconds" }

$mode = if ($NoWake)        { "no-wake (connects immediately)" } else { "wake-word on (say 'Xiaoyi' to wake)" }
$life = if ($Seconds -gt 0) { "auto-exit after $Seconds s" }     else { "runs until Ctrl+C" }
Info "2/2 Launching Xiaoyi  [$mode; $life]"
Write-Host "        Dashboard: $url" -ForegroundColor DarkGray
Write-Host "        d01 log:   $mainLog  (overwritten each run)" -ForegroundColor DarkGray
Write-Host "        Stop:      Ctrl+C here (graceful)  |  .\start_xiaoyi.ps1 stop (hard)" -ForegroundColor DarkGray
Write-Host ""

# Write the log as UTF-8 (no BOM) via StreamWriter and echo each line live.
# PS 5.1 Tee-Object only writes UTF-16LE, which git-bash/editors mis-read -> use this instead.
$sw = [System.IO.StreamWriter]::new($mainLog, $false, [System.Text.UTF8Encoding]::new($false))
$sw.AutoFlush = $true   # flush per line so the log is readable/greppable while running
try {
    & $py @d01Args 2>&1 | ForEach-Object {
        $line = if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { [string]$_ }
        $sw.WriteLine($line)   # -> UTF-8 file
        $line                  # -> console (live)
    }
}
finally {
    try { $sw.Flush(); $sw.Dispose() } catch { }
    if ($openJob) { Remove-Job -Job $openJob -Force -ErrorAction SilentlyContinue }
}
