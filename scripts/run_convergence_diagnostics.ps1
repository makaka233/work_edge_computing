param(
    [int]$Episodes = 100,
    [int]$Seconds = 300,
    [int]$ValidationEvery = 5,
    [int]$ValidationSeed = 10007,
    [int]$Seed = 7,
    [int]$TopK = 16,
    [double]$EntropyCoef = 0.001
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = (Get-Command python).Source
$statusPath = Join-Path $projectRoot "runs\convergence_diagnostics_status.csv"

Set-Location $projectRoot
"run_name,seed,status,start_time,end_time,exit_code" |
    Set-Content -Path $statusPath -Encoding utf8

$runName = "convergence_seed${Seed}_topk${TopK}"
$runDir = Join-Path $projectRoot "runs\$runName"
New-Item -ItemType Directory -Path $runDir -Force | Out-Null
$stdoutPath = Join-Path $runDir "console.log"
$stderrPath = Join-Path $runDir "console.error.log"
$startTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"$runName,$Seed,running,$startTime,," |
    Add-Content -Path $statusPath -Encoding utf8

& $pythonExe -u train.py `
    --config config/default.yaml `
    --mode neural `
    --device cuda `
    --seed $Seed `
    --agent-s-top-k-actions $TopK `
    --bc-seconds 60 `
    --bc-epochs 50 `
    --bc-max-samples 12000 `
    --episodes $Episodes `
    --seconds $Seconds `
    --agent-d-warmup-episodes $Episodes `
    --ppo-lr 0.00005 `
    --ppo-entropy-coef $EntropyCoef `
    --val-seconds $Seconds `
    --val-every $ValidationEvery `
    --val-seed $ValidationSeed `
    --val-freeze-agent-d `
    --save-every 5 `
    --run-name $runName `
    1>> $stdoutPath 2>> $stderrPath

$exitCode = $LASTEXITCODE
$endTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$status = if ($exitCode -eq 0) { "completed" } else { "failed" }
"$runName,$Seed,$status,$startTime,$endTime,$exitCode" |
    Add-Content -Path $statusPath -Encoding utf8
if ($exitCode -ne 0) {
    throw "Diagnostic run $runName failed with exit code $exitCode. See $stderrPath."
}
