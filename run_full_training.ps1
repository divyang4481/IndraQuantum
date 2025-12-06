Write-Host "Starting Full Training Pipeline (IndraQuantum)..."
Write-Host "This will run for ~50 Epochs. You can stop it with Ctrl+C and resume later."
Write-Host "Checkpoints are saved in checkpoints/full_training/"
Write-Host "Logs are saved in logs/train_full.log"

python scripts/train_full.py
