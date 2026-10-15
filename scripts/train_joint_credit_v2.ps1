param(
    [ValidateRange(1, 100000)][int]$Updates = 80,
    [ValidateRange(1, 600)][int]$SampledSeconds = 6,
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [int]$Seed = 2026,
    [int]$PhysicalSeed = 2026,
    [string]$RunName = '',
    [string]$InitFastCheckpoint = 'runs/slow_repair_scratch_s6_u320_20261014_162352/checkpoints/latest.pt',
    [ValidateSet('joint-credit-v2', 'joint-credit-v3', 'joint-credit-v4', 'joint-credit-v5')][string]$ProposedDesign = 'joint-credit-v2',
    [ValidateSet('auto', 'deployment-aware-v2', 'deployment-chain-v3')][string]$SlowDeploymentContext = 'auto',
    [string]$InitSlowCheckpoint = '',
    [ValidateSet('all', 'stage-placement')][string]$InitSlowActorScope = 'all',
    [ValidateSet(1, 3)][int]$StagePlacementLrMultiplier = 1,
    [switch]$FromScratch,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $RunName) {
    $RunName = $ProposedDesign.Replace('-', '_') + '_s' + $SampledSeconds + '_u' + $Updates + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
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
        '-u', '.\train_dual_ppo.py', '--seed', "$Seed", '--physical-seed', "$PhysicalSeed",
        '--pressure-profile', 'mec-latency200', '--proposed-design', $ProposedDesign,
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
    if ($SlowDeploymentContext -ne 'auto') {
        $trainingArguments += @('--slow-deployment-context', $SlowDeploymentContext)
    }
    if ($InitSlowCheckpoint) {
        $trainingArguments += @('--init-slow-checkpoint', $InitSlowCheckpoint, '--init-slow-actor-scope', $InitSlowActorScope)
    }
    if ($StagePlacementLrMultiplier -ne 1) {
        $trainingArguments += @('--slow-stage-lr', (0.0002 * $StagePlacementLrMultiplier).ToString([System.Globalization.CultureInfo]::InvariantCulture),
            '--slow-placement-lr', (0.0001 * $StagePlacementLrMultiplier).ToString([System.Globalization.CultureInfo]::InvariantCulture))
    }
    if ($DryRun) {
        [ordered]@{ executable='python'; working_directory=$projectRoot; arguments=$trainingArguments } | ConvertTo-Json -Depth 4
        return
    }
    & python @trainingArguments
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
