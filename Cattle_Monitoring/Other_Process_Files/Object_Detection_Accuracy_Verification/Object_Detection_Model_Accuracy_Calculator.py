# CHECK THE ACCURACY AND PERFORMANCE OF THE CURRENT OBJECT DETECTION MODEL AGAINST THE ANNOTATED LAB BOUNDING BOX DATASET

# scans labels to see what class ids you actually have
import os, glob
from ultralytics import YOLO
import pandas as pd
from pathlib import Path
import yaml, os


LABELS_DIR = r"train"
ids = set()
for fp in glob.glob(os.path.join(LABELS_DIR, "*.txt")):
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                try: 
                    id = int(float(parts[0]))
                    if id != 19:
                        print(fp, id)
                    ids.add(id)
                except: pass
print("Found class IDs:", ids)

for fp in glob.glob(os.path.join(LABELS_DIR, "*.txt")):
    out = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                p[0] = "19"  # set class to 19
                out.append(" ".join(p))
    with open(fp, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


MODEL = r"Object_Detection_Trained_Model.pt"
DATA  = r"yolo_dataset/data.yaml"

model = YOLO(MODEL)
metrics = model.val(
    data=DATA,
    split="val",
    imgsz=1280,
    conf=0.001,
    device=0,
    plots=True
)

# ---- Build Excel ----
out_xlsx = Path(DATA).with_name("val_metrics.xlsx")

# Ultralytics exposes a convenient dict of summary metrics
summary = metrics.results_dict  # e.g. metrics/mAP50(B), metrics/mAP50-95(B), precision, recall...
df_summary = pd.DataFrame(list(summary.items()), columns=["metric", "value"])

# Per-class AP (COCO mAP@0.5:0.95)
class_names = getattr(metrics, "names", getattr(model, "names", {}))
maps = getattr(metrics.box, "maps", [])  # list of floats per class index
df_perclass = pd.DataFrame({
    "class_id": list(range(len(maps))),
    "class_name": [class_names.get(i, str(i)) for i in range(len(maps))],
    "AP@0.5:0.95": maps
})

with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xw:
    df_summary.to_excel(xw, sheet_name="Summary", index=False)
    df_perclass.to_excel(xw, sheet_name="PerClass_AP", index=False)

print(f"Saved Excel -> {out_xlsx}")
print(f"Ultralytics plots/PR curves -> {metrics.save_dir}")


DATA_ROOT = r"yolo_dataset"

m = YOLO(MODEL)
names = [m.names[i] for i in range(len(m.names))]  # ensures index 19 == 'cow'

yaml_dict = {
    "path": DATA_ROOT.replace("\\", "/"),
    "train": "images/train",
    "val": "images/train",
    "names": names
}
yaml_path = Path(DATA_ROOT) / "data_coco_cow19.yaml"
with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(yaml_dict, f, sort_keys=False, allow_unicode=True)
print("Wrote:", yaml_path, "Index 19 =", names[19])

# Clear old caches (important!)
for cache in [
    os.path.join(DATA_ROOT, "labels", "train.cache"),
    os.path.join(DATA_ROOT, "labels", "val.cache"),
    os.path.join(DATA_ROOT, "images", "train.cache"),
    os.path.join(DATA_ROOT, "images", "val.cache"),
]:
    if os.path.exists(cache):
        os.remove(cache); print("Removed", cache)



MODEL = r"Object_Detection_Trained_Model.pt"
DATA  = r"yolo_dataset\data_coco_cow19.yaml"

model = YOLO(MODEL)
metrics = model.val(data=DATA, split="val", imgsz=1280, conf=0.001, device=0, plots=True, classes=[19])
print("Names used:", getattr(metrics, "names", {}))
print("mAP@0.5:", metrics.box.map50, " mAP@0.5:0.95:", metrics.box.map)




