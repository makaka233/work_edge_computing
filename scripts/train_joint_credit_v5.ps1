param(
    [ValidateRange(1, 100000)][int]$Updates = 320,
    [ValidateRange(1, 600)][int]$SampledSeconds = 6,
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [int]$Seed = 2026,
    [int]$PhysicalSeed = 2026,
    [string]$RunName = '',
    [string]$InitFastCheckpoint = '',
    [string]$InitSlowCheckpoint = '',
    [ValidateSet('all', 'stage-placement')][string]$InitSlowActorScope = 'all',
    [ValidateSet(1, 3)][int]$StagePlacementLrMultiplier = 1,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$launch = @{
    Updates=$Updates; SampledSeconds=$SampledSeconds; Device=$Device
    Seed=$Seed; PhysicalSeed=$PhysicalSeed; RunName=$RunName
    ProposedDesign='joint-credit-v5'; SlowDeploymentContext='deployment-chain-v3'
    InitFastCheckpoint=$InitFastCheckpoint; InitSlowCheckpoint=$InitSlowCheckpoint
    InitSlowActorScope=$InitSlowActorScope; StagePlacementLrMultiplier=$StagePlacementLrMultiplier
    FromScratch=(-not $InitFastCheckpoint)
    DryRun=$DryRun
}
& (Join-Path $PSScriptRoot 'train_joint_credit_v2.ps1') @launch
