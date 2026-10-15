param(
    [ValidateRange(1, 100000)][int]$Updates = 80,
    [ValidateRange(1, 600)][int]$SampledSeconds = 6,
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [string]$RunName = '',
    [string]$InitFastCheckpoint = 'runs/joint_credit_v2_s6_u320_20261015_144010/checkpoints/snapshots/update_0060.pt',
    [ValidateSet('auto', 'deployment-aware-v2', 'deployment-chain-v3')][string]$SlowDeploymentContext = 'auto',
    [string]$InitSlowCheckpoint = '',
    [ValidateSet('all', 'stage-placement')][string]$InitSlowActorScope = 'all',
    [switch]$FromScratch
)

$ErrorActionPreference = 'Stop'
$launch = @{
    Updates = $Updates; SampledSeconds = $SampledSeconds; Device = $Device
    RunName = $RunName; InitFastCheckpoint = $InitFastCheckpoint
    ProposedDesign = 'joint-credit-v3'; FromScratch = $FromScratch
    SlowDeploymentContext = $SlowDeploymentContext; InitSlowCheckpoint = $InitSlowCheckpoint
    InitSlowActorScope = $InitSlowActorScope
}
& (Join-Path $PSScriptRoot 'train_joint_credit_v2.ps1') @launch
