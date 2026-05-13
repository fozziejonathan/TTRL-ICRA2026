@echo off
setlocal EnableExtensions
REM Isaac Sim 4.5 (Isaaclab_Hover): D3D12 + RTX instance cap via CLI (reliable on hybrid laptop GPUs).

set "MINICONDA=%USERPROFILE%\miniconda3"
set "CONDA_ENV=Isaaclab_Hover"

if not exist "%MINICONDA%\Scripts\activate.bat" (
  echo ERROR: Miniconda not found at "%MINICONDA%"
  exit /b 1
)
call "%MINICONDA%\Scripts\activate.bat" "%CONDA_ENV%"
if errorlevel 1 (
  echo ERROR: conda activate "%CONDA_ENV%" failed.
  exit /b 1
)

isaacsim isaacsim.exp.base.kit ^
  --/app/vulkan=false ^
  --/rtx-transient/scenedb/maxInstancesLimit=262144 ^
  --/rtx-transient/scenedb/forceMaxTLASInstancesLimit=false ^
  %*

endlocal
