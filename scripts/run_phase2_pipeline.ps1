$ErrorActionPreference = "Stop"

Write-Host "Starting Phase 2 Pipeline..."

# 1. Precompute Logits (Critical for removing Teacher from VRAM)
# Using Batch Size 8 to ensure safety with 1024 context len on 6GB VRAM
Write-Host "Step 1: Pre-computing Teacher Logits (this will take time)..."
# Clean up old logits if needed? The script matches overwrite mode 'w+', but clean start is better.
if (Test-Path "data/precomputed_logits_v2") {
    Write-Host "Cleaning up old logits directory..."
    Remove-Item -Path "data/precomputed_logits_v2" -Recurse -Force
}

python scripts/precompute_logits.py --batch_size 8 --output_dir "data/precomputed_logits_v2" --top_k 100

if ($LASTEXITCODE -ne 0) {
    Write-Error "Pre-computation failed!"
    exit 1
}

# 2. Training
Write-Host "Step 2: Starting Training..."
python scripts/train_phase2_v4_unified.py
