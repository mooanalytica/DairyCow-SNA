param(
    [ValidateSet("auto", "cpu", "cu118")]
    [string]$TorchVariant = "auto",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectRoot "venv_windows"
$RequirementsPath = Join-Path $ProjectRoot "windows-runtime-pins.txt"

function Resolve-TorchVariant {
    param([string]$Requested)

    if ($Requested -ne "auto") {
        return $Requested
    }

    $nvidiaSmi = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
    if ($null -ne $nvidiaSmi) {
        return "cu118"
    }

    return "cpu"
}

$ResolvedVariant = Resolve-TorchVariant -Requested $TorchVariant
$TorchIndexUrl = switch ($ResolvedVariant) {
    "cu118" { "https://download.pytorch.org/whl/cu118" }
    "cpu" { "https://download.pytorch.org/whl/cpu" }
    default { throw "Unsupported TorchVariant: $ResolvedVariant" }
}

$MmcvWheelIndex = switch ($ResolvedVariant) {
    "cu118" { "https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html" }
    "cpu" { "https://download.openmmlab.com/mmcv/dist/cpu/torch2.0.0/index.html" }
    default { throw "Unsupported mmcv-full variant: $ResolvedVariant" }
}

Write-Host "[info] project root: $ProjectRoot"
Write-Host "[info] requested torch variant: $TorchVariant"
Write-Host "[info] resolved torch variant: $ResolvedVariant"

if (-not (Test-Path $VenvDir)) {
    & $PythonExe -m venv $VenvDir
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Missing venv python at $VenvPython"
}

& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPython -m pip install --index-url $TorchIndexUrl torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
& $VenvPython -m pip install -r $RequirementsPath
& $VenvPython -m pip install mmcv-full==1.5.0 -f $MmcvWheelIndex
& $VenvPython -m pip install cython_bbox

@"
from pathlib import Path
import sys
import cv2
import torch

project_root = Path(r"$ProjectRoot")
sys.path.insert(0, str(project_root / "vendor" / "ByteTrack"))
sys.path.insert(0, str(project_root / "vendor" / "ZebraPoseViTPose"))

from yolox.tracker.byte_tracker import BYTETracker
from mmpose.apis import init_pose_model

required = [
    project_root / "run_full_interaction_pipeline.py",
    project_root / "models" / "Object_Detection_Trained_Model.pt",
    project_root / "models" / "Identification_Model_Trained.pt",
    project_root / "models" / "Keypoint_Model_Trained.pth",
    project_root / "models" / "stage1_tcn_best.pt",
    project_root / "models" / "stage2_inception_best.pt",
    project_root / "models" / "svm_model.joblib",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"Missing bundled files: {missing}")

print("[ok] torch =", torch.__version__)
print("[ok] torch_cuda_available =", torch.cuda.is_available())
print("[ok] opencv =", cv2.__version__)
print("[ok] BYTETracker import = ok")
print("[ok] mmpose import = ok")
print("[ok] project files present")
"@ | & $VenvPython -

Write-Host "[ok] bootstrap complete"
Write-Host "[next] $VenvPython `"$ProjectRoot\run_full_interaction_pipeline.py`" --input-dir `"$ProjectRoot\inputs`""
