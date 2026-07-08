<#
Runs the full nuScenes mini BEVFusion batch after WSL2 is working.

Before running this script, verify:

    wsl -d Ubuntu-20.04 -- uname -a
    wsl -d Ubuntu-20.04 -- nvidia-smi

Then run:

    .\scripts\run_full_mini_after_wsl.ps1
#>

$ErrorActionPreference = "Stop"

$Project = "D:\my_project\multi-camera-lidar-bev-perception"
$WslProject = "/mnt/d/my_project/multi-camera-lidar-bev-perception"
$Micromamba = "/home/yan/.local/bin/micromamba"
$MicromambaRoot = "/home/yan/micromamba"
$EnvName = "bevfusion"

Set-Location $Project

Write-Host "Checking WSL2..." -ForegroundColor Cyan
wsl.exe -d Ubuntu-20.04 -- uname -a

Write-Host "Checking NVIDIA GPU inside WSL2..." -ForegroundColor Cyan
wsl.exe -d Ubuntu-20.04 -- nvidia-smi

Write-Host "Running BEVFusion over all nuScenes mini scenes..." -ForegroundColor Cyan
wsl.exe -d Ubuntu-20.04 --cd $WslProject env `
    PYTHONPATH=$WslProject/src `
    LD_LIBRARY_PATH=/home/yan/micromamba/envs/bevfusion/lib `
    $Micromamba run -r $MicromambaRoot -n $EnvName python scripts/run_bevfusion_batch.py `
    configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py `
    data/checkpoints/bevfusion_nuscenes.pth `
    --dataroot data/external/nuscenes `
    --infos data/external/nuscenes/nuscenes_mini_infos_all.pkl `
    --output-dir output/bevfusion_mini/scenes `
    --summary output/bevfusion_mini/batch_summary.json `
    --scene-indices all

Write-Host "Aggregating detection and tracking diagnostics..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe scripts\evaluate_bevfusion_batch.py `
    --prediction-dir output\bevfusion_mini\scenes `
    --output output\bevfusion_mini\evaluation_summary.json

Write-Host "Running unit tests..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pytest -q

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "Batch summary:     $Project\output\bevfusion_mini\batch_summary.json"
Write-Host "Evaluation report: $Project\output\bevfusion_mini\evaluation_summary.json"
