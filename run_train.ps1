param(
    [string]$Script = "scripts/3_train.py"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "venv no encontrado: $py" }

Write-Host "=== Preflight GPU ===" -ForegroundColor Cyan
& $py -c "import torch; assert torch.cuda.is_available(), 'CUDA NO disponible'; d=torch.cuda.get_device_name(0); cap=torch.cuda.get_device_capability(0); vram=round(torch.cuda.get_device_properties(0).total_memory/1024**3,2); print(f'GPU: {d} | sm_{cap[0]}{cap[1]} | VRAM {vram} GB'); x=torch.randn(2048,2048,device='cuda'); (x@x).sum().item(); print('matmul CUDA OK')"
if ($LASTEXITCODE -ne 0) { throw "Preflight GPU fallo" }

Write-Host "`n=== Ejecutando $Script ===" -ForegroundColor Cyan
& $py $Script
exit $LASTEXITCODE
