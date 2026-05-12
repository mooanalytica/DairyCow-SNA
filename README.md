# 🐄 DairyCow-SNA

AI + Computer Vision Pipeline for Social Network Analysis of Dairy Cows

## Overview

DairyCow-SNA is an open-source AI pipeline that detects, tracks, and interprets the social behavior of dairy cows using computer vision and machine learning.

It integrates YOLOv11-style detection workflows, ByteTrack, and ZebraPose-derived pose models to identify cows, estimate keypoints, and infer interactions, generating digital social profiles for each animal.

The goal is to advance precision livestock management and improve animal welfare through data-driven behavioral insights.

This repository version is a GitHub-pushable packaging of the full cattle interaction pipeline derived from `Interaction_Tracker_2.py`.

At a high level, the packaged pipeline does all of the following:

- object detection
- multi-object tracking with ByteTrack
- per-track cattle identity classification
- top-down pose inference
- pairwise interaction gating
- Stage-1 interaction / no-interaction temporal classification
- Stage-2 interaction-type classification
- export of annotated video and CSV results

The packaged project is intended to be portable across Linux and Windows.
The runtime code is included in the repository, while the large model binaries are intentionally kept outside GitHub and must be downloaded separately.

## Core Modules

| Module | Purpose |
| --- | --- |
| Object Detection Training | Trains YOLO-based models on barn environments to locate individual cows. |
| ByteTrack Optimization | Optimizes multi-object tracking with Kalman smoothing for ID consistency. |
| Object Identification | Recognizes individual cows via re-identification classification. |
| ZebraPose Keypoint Detection | Detects cow body keypoints for posture and movement tracking. |
| YOLO Keypoint Detection | Alternate pose model path for custom cow keypoint detection workflows. |
| Interaction Inference | Uses temporal keypoint distances and learned classifiers to identify interaction types such as affiliative, neutral, and aggressive behavior. |
| Cattle Monitoring Main Pipeline | Integrates all modules and produces downstream behavioral and social-network-ready outputs. |

## Workflow

Detection -> Tracking -> Identification -> Keypoint Detection -> Interaction Inference -> Social Network Graphs

Each cow becomes a node in a dynamic social graph, and interactions form edges weighted by frequency and duration.

## What Is Included

Project root contents:

- `run_full_interaction_pipeline.py`
  The main runner for the full interaction pipeline.
- `inputs/`
  Default location for input videos.
- `outputs/`
  Default location for output run folders.
- `models/`
  Required model directory. In this GitHub-pushable version, the large model files are not committed and must be downloaded separately.
- `Beyond_Proximity_Dataset/`
  Reference folder for the public sample dataset download location and expected structure.
- `vendor/ByteTrack/`
  Vendored ByteTrack runtime code.
- `vendor/ZebraPoseViTPose/`
  Vendored pose runtime code and pose config files.
- `bootstrap_linux.sh`
  Linux environment bootstrap.
- `bootstrap_windows.ps1`
  Windows environment bootstrap.
- `requirements.txt`
  Lightweight root requirements for CI, dependency submission, and smoke tests.
- `linux-runtime-pins.txt`
  Linux runtime dependency pins used by `bootstrap_linux.sh`.
- `windows-runtime-pins.txt`
  Windows runtime dependency pins used by `bootstrap_windows.ps1`.

## Model Weights

Model files are stored externally due to GitHub size limits.

This packaged project expects the following required files under `models/`:

- `Object_Detection_Trained_Model.pt`
- `Identification_Model_Trained.pt`
- `Keypoint_Model_Trained.pth`
- `svm_model.joblib`
- `stage1_tcn_best.pt`
- `stage2_inception_best.pt`

These files are not stored in the GitHub-pushable version of the repository.

Download them from this Google Drive folder:

https://drive.google.com/drive/folders/1im99sooJqAi70oieGIO0xhpFePfS0qoD?usp=sharing

After downloading, place them directly into the `models/` folder.

## Sample Dataset

Sample data from the Beyond Proximity dataset is also stored externally and is not committed into this GitHub repository.

Download the sample dataset from this Google Drive folder:

https://drive.google.com/drive/folders/1sP_RqsjEJuXkHw8UJ1wP3fz_waFQeCyj?usp=sharing

The external sample dataset currently contains these top-level directories:

- `Interaction_Feature_Keypoint`
- `Keypoint_Detection`
- `Videos`

If you want the local project layout to mirror the referenced dataset name, place the downloaded contents under:

- `./Beyond_Proximity_Dataset/Interaction_Feature_Keypoint`
- `./Beyond_Proximity_Dataset/Keypoint_Detection`
- `./Beyond_Proximity_Dataset/Videos`

The repository includes a placeholder `Beyond_Proximity_Dataset/README.md` so the sample dataset is clearly referenced inside GitHub without storing the data itself in version control.

## Default Project Behavior

If you run the script with no overrides beyond `--input-dir`, it uses project-relative defaults:

- input videos: `./inputs`
- output base directory: `./outputs`
- detector weights: `./models/Object_Detection_Trained_Model.pt`
- identity weights: `./models/Identification_Model_Trained.pt`
- pose checkpoint: `./models/Keypoint_Model_Trained.pth`
- SVM model: `./models/svm_model.joblib`
- Stage-1 checkpoint: `./models/stage1_tcn_best.pt`
- Stage-2 checkpoint: `./models/stage2_inception_best.pt`
- pose config:
  `./vendor/ZebraPoseViTPose/ZebraPose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/MAE_pret_syn/s_zebras_old_adam.py`

Each run creates a new numbered output folder inside `outputs/`, for example:

- `outputs/1`
- `outputs/2`
- `outputs/3`

## Supported Input Video Formats

The runner accepts these file extensions:

- `.mp4`
- `.mkv`
- `.mov`
- `.avi`
- `.m4v`
- `.wmv`
- `.ts`
- `.webm`

By default, the runner scans the chosen input directory and processes all supported video files found there.

## Quick Start

The normal workflow is:

1. Download the model files into `models/`.
2. Optionally download the public sample dataset into `Beyond_Proximity_Dataset/`.
3. Set up the Python environment.
4. Put videos into `inputs/` or pass specific video paths with `--video`.
5. Run `run_full_interaction_pipeline.py`.
6. Read the results from a new numbered folder under `outputs/`.

For lightweight setup, GitHub dependency submission, and reviewer-safe smoke tests, install the root `requirements.txt`.

For the full packaged pipeline environment, use the platform bootstrap scripts, which install `linux-runtime-pins.txt` or `windows-runtime-pins.txt` plus the matching Torch and `mmcv-full` wheels for the selected platform.

## Linux Setup

### What The Linux Bootstrap Does

`bootstrap_linux.sh` is a general Linux bootstrap script.

It:

- creates a virtual environment in `./venv`
- installs Python packaging tools
- installs PyTorch
- installs runtime dependencies from `linux-runtime-pins.txt`
- installs `mmcv-full==1.5.0` from an OpenMMLab wheel index chosen to match the selected Torch variant
- installs `cython_bbox`
- verifies that the required project files and imports are present

When `apt-get` is available, it also installs these system packages:

- `ffmpeg`
- `libgl1`
- `libglib2.0-0`
- `build-essential`
- `python3-dev`
- `python3-venv`
- `python3-pip`
- `git`

If `apt-get` is not available, the script prints the packages you should install manually before continuing.

### Linux GPU / CPU Selection

The Linux bootstrap uses `TORCH_VARIANT`.

Supported values:

- `auto`
- `cpu`
- `cu118`

Behavior:

- `auto` selects `cu118` when `nvidia-smi` is available
- `auto` selects `cpu` when no NVIDIA GPU is detected

### Linux Commands

GPU-or-auto setup:

```bash
bash bootstrap_linux.sh
source venv/bin/activate
python run_full_interaction_pipeline.py --input-dir ./inputs
```

Force CPU setup:

```bash
TORCH_VARIANT=cpu bash bootstrap_linux.sh
source venv/bin/activate
python run_full_interaction_pipeline.py --input-dir ./inputs
```

Force CUDA 11.8 setup:

```bash
TORCH_VARIANT=cu118 bash bootstrap_linux.sh
source venv/bin/activate
python run_full_interaction_pipeline.py --input-dir ./inputs
```

## Windows Setup

### What The Windows Bootstrap Does

`bootstrap_windows.ps1` is the Windows environment bootstrap.

It:

- creates a virtual environment in `.\venv_windows`
- installs PyTorch
- installs runtime dependencies from `windows-runtime-pins.txt`
- installs `mmcv-full==1.5.0` from an OpenMMLab wheel index chosen to match the selected Torch variant
- installs `cython_bbox`
- verifies that the required project files and imports are present

### Windows GPU / CPU Selection

The Windows bootstrap uses `-TorchVariant`.

Supported values:

- `auto`
- `cpu`
- `cu118`

Behavior:

- `auto` selects `cu118` when `nvidia-smi.exe` is available
- `auto` selects `cpu` otherwise

### Windows Commands

From PowerShell in the project root:

Auto-select CPU or GPU:

```powershell
.\bootstrap_windows.ps1
.\venv_windows\Scripts\python.exe .\run_full_interaction_pipeline.py --input-dir .\inputs
```

Force CPU:

```powershell
.\bootstrap_windows.ps1 -TorchVariant cpu
.\venv_windows\Scripts\python.exe .\run_full_interaction_pipeline.py --input-dir .\inputs
```

Force CUDA 11.8:

```powershell
.\bootstrap_windows.ps1 -TorchVariant cu118
.\venv_windows\Scripts\python.exe .\run_full_interaction_pipeline.py --input-dir .\inputs
```

If PowerShell script execution is blocked, you can run:

```powershell
powershell -ExecutionPolicy Bypass -File .\bootstrap_windows.ps1
```

## Running The Project

### Default Folder-Based Run

Put videos into `inputs/` and run:

Linux:

```bash
python run_full_interaction_pipeline.py --input-dir ./inputs
```

Windows:

```powershell
.\venv_windows\Scripts\python.exe .\run_full_interaction_pipeline.py --input-dir .\inputs
```

### Specific Files Instead Of A Folder

Linux:

```bash
python run_full_interaction_pipeline.py \
  --video /abs/path/video1.mp4 \
  --video /abs/path/video2.mp4
```

Windows:

```powershell
.\venv_windows\Scripts\python.exe .\run_full_interaction_pipeline.py `
  --video .\inputs\video1.mp4 `
  --video .\inputs\video2.mp4
```

### Custom Output Directory

Linux:

```bash
python run_full_interaction_pipeline.py --input-dir ./inputs --output-dir ./my_outputs
```

Windows:

```powershell
.\venv_windows\Scripts\python.exe .\run_full_interaction_pipeline.py --input-dir .\inputs --output-dir .\my_outputs
```

## CLI Reference

The runner currently exposes these command-line options:

### `--input-dir`

- Type: path
- Default: `./inputs`
- Meaning: directory scanned for supported video files

### `--video`

- Type: path, repeatable
- Default: none
- Meaning: process one or more explicitly named video files
- Note: you can repeat `--video` multiple times

### `--output-dir`

- Type: path
- Default: `./outputs`
- Meaning: base directory where numbered run folders are created

### `--det-weights`

- Type: path
- Default: `./models/Object_Detection_Trained_Model.pt`
- Meaning: detector weights override

### `--id-weights`

- Type: path
- Default: `./models/Identification_Model_Trained.pt`
- Meaning: identity model weights override

### `--pose-config`

- Type: path
- Default:
  `./vendor/ZebraPoseViTPose/ZebraPose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/MAE_pret_syn/s_zebras_old_adam.py`
- Meaning: pose config override

### `--pose-checkpoint`

- Type: path
- Default: `./models/Keypoint_Model_Trained.pth`
- Meaning: pose checkpoint override

### `--svm-model`

- Type: path
- Default: `./models/svm_model.joblib`
- Meaning: optional SVM model override

### `--stage1-ckpt`

- Type: path
- Default: `./models/stage1_tcn_best.pt`
- Meaning: Stage-1 temporal checkpoint override

### `--stage2-ckpt`

- Type: path
- Default: `./models/stage2_inception_best.pt`
- Meaning: Stage-2 temporal checkpoint override

### `--det-device`

- Type: string or GPU index
- Default: `auto`
- Supported examples: `auto`, `0`, `cpu`
- Meaning: detector and identity model device selection

### `--pose-device`

- Type: string
- Default: `auto`
- Supported examples: `auto`, `cuda:0`, `cpu`
- Meaning: pose model device selection

### `--annotated-output-scale`

- Type: float
- Default: `0.5`
- Meaning: scaling factor applied to the annotated output video size

### `--disable-annotated-video`

- Type: flag
- Default: off
- Meaning: disables annotated video writing to reduce runtime and output size

### Built-In Auto Device Behavior

When `auto` is used:

- detector device becomes GPU `0` when CUDA is available, otherwise `cpu`
- pose device becomes `cuda:0` when CUDA is available, otherwise `cpu`

## Outputs

For each run, the script creates a new numbered folder under `outputs/` or the directory provided by `--output-dir`.

Typical outputs include:

### `tracking_boxes.csv`

One row per track per frame, including:

- video name
- frame index
- track ID
- bounding box
- tracking score
- resolved identity label
- identity confidence

### `interactions.csv`

Detected interaction events, including:

- video
- interaction class
- track IDs for the pair
- start and end frame
- duration
- event weight
- Stage-2 confidence
- Stage-1 probability
- Stage-1 threshold

### `no_interactions.csv`

Candidate overlapping pairs that remained negative for interaction.

### `proximity_percentiles.csv`

Saved when proximity logging is enabled in the script.

### `adjacency_<class>.csv`

Per-class adjacency summaries built from the detected interactions.

### `*_tracked_pose.mp4`

Annotated output video, unless `--disable-annotated-video` is used.

### `keypoints.csv`

The code supports this file, but it is disabled by default because `SAVE_KEYPOINTS_CSV = False` in the runner.

## Referenced External Assets

The repository intentionally references, rather than stores, two large external asset bundles:

- model weights:
  https://drive.google.com/drive/folders/1im99sooJqAi70oieGIO0xhpFePfS0qoD?usp=sharing
- sample Beyond Proximity dataset:
  https://drive.google.com/drive/folders/1sP_RqsjEJuXkHw8UJ1wP3fz_waFQeCyj?usp=sharing

This keeps the GitHub repository lightweight while still documenting how to reconstruct the runnable environment and sample-data layout.

## Important Runtime Notes

- The runner creates a fresh numbered output folder on each run instead of overwriting the last run automatically.
- Annotated video output is enabled by default.
- Keypoint CSV export is disabled by default.
- The script processes all supported videos in the selected input directory unless you explicitly use `--video`.
- Stage-2 confidence gating is enabled in the runner.
- The pipeline includes cooldown and temporal stability logic for interaction events, so outputs are not simple frame-by-frame raw detections.

## Known Assumptions And Limitations

- This packaged project was assembled from the existing working code and assets, but I did not run the full pipeline end-to-end in this packaged form during this session.
- The dependency stacks in `bootstrap_linux.sh` and `bootstrap_windows.ps1` are best reconstructed environment definitions, not yet fully validated handoff environments.
- GPU support is intentionally left open on both Linux and Windows, but the exact CUDA / driver / wheel compatibility still depends on the target machine.
- The Windows path is more manual and riskier than the Linux path because packages like `mmcv-full` can be sensitive to Python, Torch, and platform combinations.
- The vendored pose stack is older and depends on the matching `mmcv-full` / Torch family specified by the bootstrap scripts.

## Data Policy

Full cow video datasets are restricted and are not published online in this repository.

- only project code, configuration, and lightweight examples should be shared here
- model files are hosted externally due to GitHub size limits
- the publicly shared Beyond Proximity sample dataset is referenced externally through Google Drive rather than committed into Git
- any third-party or human data from external sources should remain removed for copyright and privacy compliance

## Team

- Lead Developer: Sibi Parivendhan
- Supervisor: Dr. Suresh Raja Neethirajan, Dalhousie University

## License

This project is licensed under the Apache 2.0 License.

See the `LICENSE` file for details if it is included in the target repository.

## Research Links

- MooAnalytica Research Group: https://mooanalytica.com
- Dalhousie Faculty of Agriculture and Computer Science:
  https://www.dal.ca/faculty/computerscience/faculty-staff/Suresh-Raja-Neethirajan.html

## Citation

If you use this work, please cite:

S. Parivendan, K. Sailunaz, S. Neethirajan (2025). DairyCow-SNA: AI-Enabled Social Network Analysis of Dairy Cows.

## Recommended First Validation

Before running a large batch, do a quick smoke test:

1. Put one short input video into `inputs/`.
2. Run the bootstrap for your platform.
3. Run the main script once.
4. Check that a numbered run folder appears under `outputs/`.
5. Confirm that `tracking_boxes.csv`, `interactions.csv` or `no_interactions.csv`, and an annotated video are produced as expected.

## Minimal Everyday Usage Summary

If the environment is already set up, the common usage is exactly this:

1. Download the required model files into `models/`.
2. Optionally download the public sample dataset into `Beyond_Proximity_Dataset/`.
3. Put videos into `inputs/` or select files from the downloaded sample dataset.
4. Run `run_full_interaction_pipeline.py`.
5. Read the results from the newest numbered folder under `outputs/`.
