param(
    [int]$Seed = 1234,
    [int]$Epochs = 2,
    [int]$TrainLimit = 5000,
    [int]$ValidationLimit = 1000
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repositoryRoot

try {
    Remove-Item -Force logs\experiments.csv -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force weights\tuning -ErrorAction SilentlyContinue

    # A small grid over hidden-layer size, learning rates, and epochs.
    # Runs use a subset of the training data so the sweep finishes quickly;
    # the winning configuration is then retrained on the full 50k split.
    $configurations = @(
        @{ Hidden = 32;  Lr = 0.10; HiddenLr = 0.10; Epochs = $Epochs },
        @{ Hidden = 64;  Lr = 0.10; HiddenLr = 0.10; Epochs = $Epochs },
        @{ Hidden = 64;  Lr = 0.20; HiddenLr = 0.15; Epochs = $Epochs },
        @{ Hidden = 64;  Lr = 0.30; HiddenLr = 0.20; Epochs = $Epochs },
        @{ Hidden = 128; Lr = 0.10; HiddenLr = 0.10; Epochs = $Epochs },
        @{ Hidden = 128; Lr = 0.20; HiddenLr = 0.15; Epochs = $Epochs },
        @{ Hidden = 64;  Lr = 0.20; HiddenLr = 0.15; Epochs = 1 },
        @{ Hidden = 64;  Lr = 0.20; HiddenLr = 0.15; Epochs = 4 }
    )

    foreach ($configuration in $configurations) {
        Write-Host "=== hidden=$($configuration.Hidden) lr=$($configuration.Lr) hidden-lr=$($configuration.HiddenLr) epochs=$($configuration.Epochs) ==="
        dotnet run -- --hidden $configuration.Hidden --lr $configuration.Lr --hidden-lr $configuration.HiddenLr --epochs $configuration.Epochs --seed $Seed --train-limit $TrainLimit --val-limit $ValidationLimit --weight-dir weights/tuning --summary-csv logs/experiments.csv --skip-test
        if ($LASTEXITCODE -ne 0) {
            throw "Training run failed with exit code $LASTEXITCODE."
        }
    }

    Write-Host "Tuning complete. Results are in logs/experiments.csv"
    Remove-Item -Recurse -Force weights\tuning -ErrorAction SilentlyContinue
}
finally {
    Pop-Location
}
