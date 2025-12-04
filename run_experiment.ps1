Write-Host "Starting IndraQuantum Benchmark Experiment..."

Write-Host "1. Training Efficient IndraQuantum (TT-Decomposed)..."
python scripts/train_compare.py --model quantum --epochs 1
if ($LASTEXITCODE -ne 0) { Write-Error "Quantum training failed"; exit 1 }

Write-Host "2. Training Standard Baseline..."
python scripts/train_compare.py --model baseline --epochs 1
if ($LASTEXITCODE -ne 0) { Write-Error "Baseline training failed"; exit 1 }

Write-Host "3. Training IndraQuantum Graph Model..."
python scripts/train_graph.py --epochs 1
if ($LASTEXITCODE -ne 0) { Write-Error "Graph training failed"; exit 1 }

Write-Host "4. Generating Analysis Notebook..."
python scripts/generate_notebook.py
if ($LASTEXITCODE -ne 0) { Write-Error "Notebook generation failed"; exit 1 }

Write-Host "Experiment Complete! Please open 'notebooks/Architecture_Comparison.ipynb' to view results."
