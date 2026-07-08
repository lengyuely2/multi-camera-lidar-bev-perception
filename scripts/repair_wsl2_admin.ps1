#Requires -RunAsAdministrator

<#
Restores the Windows virtualization pieces required by WSL2.

Run this script from an elevated PowerShell window:

    Set-ExecutionPolicy -Scope Process Bypass -Force
    .\scripts\repair_wsl2_admin.ps1

After it finishes, reboot Windows before starting WSL again.
#>

$ErrorActionPreference = "Stop"

Write-Host "Enabling WSL and Virtual Machine Platform..." -ForegroundColor Cyan
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -All -NoRestart
Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -All -NoRestart

Write-Host "Enabling Windows Hypervisor Platform..." -ForegroundColor Cyan
Enable-WindowsOptionalFeature -Online -FeatureName HypervisorPlatform -All -NoRestart

Write-Host "Setting the Windows hypervisor to launch automatically..." -ForegroundColor Cyan
bcdedit /set hypervisorlaunchtype auto

Write-Host "Updating WSL package/kernel..." -ForegroundColor Cyan
wsl --update

Write-Host ""
Write-Host "Done. Please reboot Windows, then verify with:" -ForegroundColor Green
Write-Host "    wsl -d Ubuntu-20.04 -- uname -a"
Write-Host "    wsl -d Ubuntu-20.04 -- nvidia-smi"
