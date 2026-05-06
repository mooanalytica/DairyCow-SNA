# Models Folder

The model binaries are intentionally not included in this GitHub-pushable version of the project.

Download the required model files from this Google Drive folder:

https://drive.google.com/drive/folders/1im99sooJqAi70oieGIO0xhpFePfS0qoD?usp=sharing

Place the downloaded files directly into this `models/` folder so the final layout becomes:

- `models/Object_Detection_Trained_Model.pt`
- `models/Identification_Model_Trained.pt`
- `models/Keypoint_Model_Trained.pth`
- `models/svm_model.joblib`
- `models/stage1_tcn_best.pt`
- `models/stage2_inception_best.pt`

The bootstrap scripts and the main pipeline runner expect those exact filenames.
