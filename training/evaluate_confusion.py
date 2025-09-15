"""Generate confusion matrix and classification report for the latest checkpoint.

Usage (examples):
  python training/evaluate_confusion.py --data dataset_clean/val
  python training/evaluate_confusion.py --data dataset_clean/val --runs runs --device cuda

Outputs (written next to run directory or into --out):
  confusion_matrix.png             – surowa macierz (liczności)
  confusion_matrix_normalized.png  – macierz znormalizowana (procenty w wierszu)
  classification_report.json       – precision / recall / f1 per klasa + makra
  per_sample_predictions.csv       – (opcjonalnie) lista: path,target,pred

Wymaga: scikit-learn, seaborn, matplotlib.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple, List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


def find_latest_checkpoint(runs_dir: str) -> Tuple[str | None, str | None]:
    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        return None, None
    candidates: List[Path] = [p for p in runs_path.iterdir() if p.is_dir()]
    if not candidates:
        return None, None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for run in candidates:
        # prefer final.pth, then best.pth, then any checkpoint
        final_pth = run / "final.pth"
        best_pth = run / "best.pth"
        if final_pth.is_file():
            return str(final_pth), str(run / "label_map.json")
        if best_pth.is_file():
            return str(best_pth), str(run / "label_map.json")
        ckpts = sorted(run.glob("checkpoint_epoch_*.pth"))
        if ckpts:
            return str(ckpts[-1]), str(run / "label_map.json")
    return None, None


def load_classifier(model_path: str, label_map_path: str | None, device: str) -> Tuple[torch.nn.Module, Dict[int, str]]:
    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint if isinstance(checkpoint, dict) else {})
    # label map
    idx_to_label: Dict[int, str] = {}
    if label_map_path and os.path.isfile(label_map_path):
        try:
            with open(label_map_path, "r", encoding="utf-8") as f:
                label_map = json.load(f)
            # label_map może być {class_name: idx}
            # odwracamy jeśli wartości są int
            rev = {}
            for k, v in label_map.items():
                try:
                    rev[int(v)] = str(k)
                except Exception:
                    # może w formacie {name: index} -> powyższe wciąż zadziała
                    pass
            if rev:
                idx_to_label = rev
            else:
                # albo {idx: name}
                try:
                    idx_to_label = {int(k): str(v) for k, v in label_map.items()}
                except Exception:
                    pass
        except Exception:
            pass
    # liczba klas – z warstwy wyjściowej
    if isinstance(state, dict) and "classifier.1.weight" in state:
        num_classes = state["classifier.1.weight"].shape[0]
    else:
        num_classes = len(idx_to_label) if idx_to_label else 5
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        # permissive load
        model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
    model.eval()
    model.to(device)
    return model, idx_to_label


def build_loader(data_dir: str, img_size: int, batch: int, workers: int) -> Tuple[DataLoader, List[str]]:
    tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(data_dir, transform=tf)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    return loader, ds.classes


def generate_confusion(model, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(y.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def plot_and_save(cm: np.ndarray, labels: List[str], out_png: str, normalize: bool = False) -> None:
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        data = cm_norm
        fmt = ".2f"
        title = "Confusion Matrix (normalized)"
    else:
        data = cm
        fmt = "d"
        title = "Confusion Matrix"
    plt.figure(figsize=(1 + 0.7 * len(labels), 1 + 0.7 * len(labels)))
    sns.heatmap(data, annot=True, cmap="Blues", fmt=fmt, xticklabels=labels, yticklabels=labels, cbar=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Ścieżka do katalogu walidacyjnego/testowego (ImageFolder)")
    ap.add_argument("--runs", default="runs", help="Katalog z checkpointami")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None, help="Opcjonalny katalog wyjściowy (domyślnie obok checkpointu)")
    ap.add_argument("--per-sample-csv", action="store_true", help="Zapisz per_sample_predictions.csv")
    args = ap.parse_args()

    ckpt, label_map_path = find_latest_checkpoint(args.runs)
    if not ckpt:
        raise SystemExit(f"Nie znaleziono checkpointów w '{args.runs}'")
    print(f"Używam checkpointu: {ckpt}")
    model, idx_to_label = load_classifier(ckpt, label_map_path, args.device)
    loader, folder_labels = build_loader(args.data, args.img_size, args.batch, args.workers)

    # Mapy nazw: preferuj folder_labels dla spójności z ground truth
    label_names = folder_labels

    y_true, y_pred = generate_confusion(model, loader, args.device)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    print("Macierz kształt:", cm.shape)

    # Ścieżki wyjściowe
    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = Path(ckpt).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_and_save(cm, label_names, str(out_dir / "confusion_matrix.png"), normalize=False)
    plot_and_save(cm, label_names, str(out_dir / "confusion_matrix_normalized.png"), normalize=True)
    print("Zapisano confusion_matrix*.png w", out_dir)

    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0)
    with open(out_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Zapisano classification_report.json")

    if args.per_sample_csv:
        import csv
        csv_path = out_dir / "per_sample_predictions.csv"
        # Aby pozyskać ścieżki obrazów, przebuduj loader.dataset.samples
        samples = getattr(loader.dataset, "samples", [])
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "true_idx", "true_label", "pred_idx", "pred_label"])
            for (path, true_idx), pred_idx in zip(samples, y_pred):
                true_label = label_names[true_idx] if true_idx < len(label_names) else str(true_idx)
                pred_label = label_names[pred_idx] if pred_idx < len(label_names) else str(pred_idx)
                w.writerow([path, true_idx, true_label, pred_idx, pred_label])
        print("Zapisano per_sample_predictions.csv")


if __name__ == "__main__":
    main()
