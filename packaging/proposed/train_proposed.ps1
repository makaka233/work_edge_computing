param(
    [int]$Updates = 320,
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [string]$RunName = '',
    [string]$LoadCheckpoint = '',
    [string]$InitFastCheckpoint = '',
    [switch]$AppendLog
)

$ErrorActionPreference = 'Stop'
if ($Updates -lt 1) { throw 'Updates must be positive' }
if ($AppendLog -and -not $RunName) { throw 'AppendLog requires the original RunName to preserve its directory' }
if (-not $RunName) { $RunName = 'proposed_joint_s6_u' + $Updates + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') }
if ($AppendLog -and -not $LoadCheckpoint) { throw 'AppendLog requires an explicit checkpoint' }
if ($LoadCheckpoint -and $InitFastCheckpoint) { throw 'LoadCheckpoint and InitFastCheckpoint are mutually exclusive' }
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not $AppendLog -and (Test-Path -LiteralPath (Join-Path 'runs' $RunName))) {
        throw 'Run directory already exists. Choose a new RunName, or explicitly resume with AppendLog.'
    }
    $trainingArguments = @(
        '.\train_dual_ppo.py', '--seed', '2026', '--pressure-profile', 'mec-latency200',
        '--training-design', 'trajectory-simultaneous', '--train-mode', 'joint', '--rollout-unit', 'window',
        '--updates', "$Updates", '--episode-minutes', '60', '--deployment-interval-minutes', '10',
        '--fast-windows-per-update', '12', '--slow-windows-per-update', '12',
        '--sampled-seconds-per-window', '6', '--sampling-max-interval-seconds', '100',
        '--num-users', '12000', '--num-edge-nodes', '32', '--num-service-types', '10',
        '--network-constraint-mode', 'shared-nic', '--service-calibration-mode', 'audited-v1',
        '--temporal-sampling-mode', 'stratified-jitter',
        '--replicas-per-stage', '0', '--reward-mode', 'latency', '--reward-scale', '10',
        '--fast-policy-kind', 'gat_node_scorer', '--fast-chain-context', 'chain-kkt-v3',
        '--fast-counterfactual-credit-coef', '0', '--fast-counterfactual-credit-final-coef', '0',
        '--slow-counterfactual-credit-coef', '0', '--slow-replica-ablation-samples', '0',
        '--fast-oracle-diagnostic-requests', '0', '--fast-k-epochs', '4', '--slow-k-epochs', '2',
        '--slow-stage-k-epochs', '3', '--slow-placement-k-epochs', '3',
        '--fast-minibatch-size', '512', '--eval-interval', '0', '--device', $Device,
        '--run-name', $RunName, '--progress-interval-seconds', '10'
    )
    if ($LoadCheckpoint) {
        $trainingArguments += @('--proposed-design', 'auto', '--load-checkpoint', $LoadCheckpoint)
    } else {
        $trainingArguments += @('--proposed-design', 'joint-credit-v2')
    }
    if ($InitFastCheckpoint) {
        $trainingArguments += @('--init-fast-checkpoint', $InitFastCheckpoint, '--init-fast-actor-only')
    }
    if ($AppendLog) { $trainingArguments += '--append-log' }
    & python @trainingArguments
    if ($LASTEXITCODE -ne 0) { throw "Training stopped with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
