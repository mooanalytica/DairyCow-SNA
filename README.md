# 🐄 DairyCow-SNA  
**AI + Computer Vision Pipeline for Social Network Analysis of Dairy Cows**

---

### 🌟 Overview
**DairyCow-SNA** is an open-source AI pipeline that detects, tracks, and interprets the social behavior of dairy cows using computer vision and machine learning.  
It integrates **YOLOv11**, **ByteTrack**, and **ZebraPose** models to identify cows, estimate keypoints, and infer interactions — generating digital social profiles for each animal.  
The goal is to advance **precision livestock management** and improve **animal welfare** through data-driven behavioral insights.

---

### 🔬 Core Modules
| Module | Purpose |
|---------|----------|
| **Object Detection Training** | Trains YOLO models on barn environments to locate individual cows. |
| **ByteTrack Optimization** | Optimizes multi-object tracking with Kalman smoothing for ID consistency. |
| **Object Identification** | Recognizes individual cows via re-ID classification. |
| **ZebraPose Keypoint Detection** | Detects cow body keypoints for posture and movement tracking. |
| **YOLO Keypoint Detection** | Alternate pose model (YOLOv11x-Pose) trained on custom cow datasets. |
| **Interaction Inference** | Uses temporal keypoint distances and SVM classifiers to identify interaction types (affiliative, neutral, aggressive). |
| **Cattle Monitoring Main Pipeline** | Integrates all modules and produces social network graphs. |

---

### ⚙️ Workflow
Detection → Tracking → Identification → Keypoint Detection → Interaction Inference → Social Network Graphs

Each cow becomes a **node** in a dynamic social graph, and interactions form **edges** weighted by frequency and duration.

---

### 🚀 Quick Start

#### 1. Clone and set up
```bash
git clone https://github.com/mooanalytica/DairyCow-SNA.git
cd DairyCow-SNA
pip install -r requirements.txt
--
Run demo
python src/pipeline_main/run_demo.py --config configs/demo.yaml

--
View outputs

Sample results (confusion matrices, precision–recall curves, prediction grids) are in
docs/example_outputs/
--
Model Weights

Download pretrained models from the Releases
 page:

yolo11x-pose_trained.pt — Keypoint detection

ours_320.pth — ZebraPose model

cowID_cls_224.pt — Re-identification model

(Model files are stored externally due to GitHub size limits.)
---
Data Policy

Full cow video datasets are restricted and not published online.
Only sample images are included for demonstration.
Any third-party or human data from external sources has been removed for copyright compliance.
---
Team

Lead Developer: Sibi Parivendhan
Supervisor: Dr. Suresh Raja Neethirajan, Dalhousie University
--
License
This project is licensed under the Apache 2.0 License.
See the LICENSE file for details.
--
Links
🔗 MooAnalytica Research Group - https://mooanalytica.com
🎓 Dalhousie Faculty of Agriculture & Computer Science - https://www.dal.ca/faculty/computerscience/faculty-staff/Suresh-Raja-Neethirajan.html


