# Car Recognition GUI (YOLO + MobileNetV2)

A PyQt5 desktop app that detects cars with YOLOv8, crops the car region, classifies brand with a MobileNetV2 classifier (from `runs/*/final.pth`), stores results in SQLite, and can export history to PDF.

## Features
- Load image → detect car (YOLOv8) → crop → classify brand (MobileNetV2)
- Grad-CAM heatmap overlay for explainability
- History stored in SQLite (`car_history.db`) and copied images in hidden `.db_images/`
- History viewer with preview and PDF export (single, checked, all)

## Requirements
- Python 3.10+
- GPU optional (CUDA). Falls back to CPU.

Install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python GUI.py
```
Place your classifier weights and label map under `runs/<any_name>/{final.pth,label_map.json}`.

## Notes
- YOLOv8 weights `yolov8s.pt` are expected in repo root (adjust in `GUI.py` if needed).
- DB and copied images are not committed (`.gitignore`).

## Next steps
- Add tests (pytest) for DB and PDF export
- CI workflow (GitHub Actions) for lint + tests
- Package the app via PyInstaller for distribution
