param(
    [ValidateRange(1, 100000)][int]$Updates = 80,
    [ValidateRange(1, 600)][int]$SampledSeconds = 6,
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [string]$RunName = '',
    [string]$InitFastCheckpoint = 'runs/slow_repair_scratch_s6_u320_20261014_162352/checkpoints/latest.pt',
    [switch]$FromScratch
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $RunName) {
    $RunName = 'joint_credit_v2_s' + $SampledSeconds + '_u' + $Updates + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
}
Push-Location -LiteralPath $projectRoot
try {
    if (Test-Path -LiteralPath (Join-Path 'runs' $RunName)) {
        throw 'Run directory already exists. Use a new RunName; this helper never overwrites a run.'
    }
    if (-not $FromScratch -and -not (Test-Path -LiteralPath $InitFastCheckpoint -PathType Leaf)) {
        throw 'Initialization checkpoint not found. Specify InitFastCheckpoint, or choose FromScratch.'
    }
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $trainingArguments = @(
        '-u', '.\train_dual_ppo.py', '--seed', '2026',
        '--pressure-profile', 'mec-latency200', '--proposed-design', 'joint-credit-v2',
        '--training-design', 'trajectory-simultaneous', '--rollout-unit', 'window',
        '--updates', "$Updates", '--episode-minutes', '60', '--deployment-interval-minutes', '10',
        '--fast-windows-per-update', '12', '--slow-windows-per-update', '12',
        '--sampled-seconds-per-window', "$SampledSeconds", '--sampling-max-interval-seconds', '100',
        '--num-users', '12000', '--num-edge-nodes', '32', '--num-service-types', '10',
        '--replicas-per-stage', '0', '--reward-mode', 'latency', '--reward-scale', '10',
        '--fast-k-epochs', '4', '--slow-k-epochs', '2',
        '--slow-stage-k-epochs', '3', '--slow-placement-k-epochs', '3',
        '--fast-minibatch-size', '512', '--eval-interval', '0', '--device', $Device,
        '--checkpoint-probe-interval', '10', '--run-name', $RunName,
        '--progress-interval-seconds', '10'
    )
    if (-not $FromScratch) {
        $trainingArguments += @('--init-fast-checkpoint', $InitFastCheckpoint, '--init-fast-actor-only')
    }
    & python @trainingArguments
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
