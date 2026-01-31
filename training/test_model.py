import os
import sys
import torch
from torchvision import models, transforms, datasets
from PIL import Image
import json
from pathlib import Path
from collections import defaultdict

def find_latest_run_checkpoint(runs_dir='runs'):
    runs_dir = os.path.abspath(runs_dir)
    if not os.path.isdir(runs_dir):
        return None, None
    candidates = []
    for name in os.listdir(runs_dir):
        p = os.path.join(runs_dir, name)
        if os.path.isdir(p):
            candidates.append(p)
    if not candidates:
        return None, None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for run in candidates:
        best = os.path.join(run, 'best.pth')
        final = os.path.join(run, 'final.pth')
        if os.path.isfile(best):
            return best, os.path.join(run, 'label_map.json')
        if os.path.isfile(final):
            return final, os.path.join(run, 'label_map.json')
        for fname in os.listdir(run):
            if fname.startswith('checkpoint_epoch_') and fname.endswith('.pth'):
                return os.path.join(run, fname), os.path.join(run, 'label_map.json')
    return None, None

def load_model(model_path, label_map_path=None, device='cpu'):
    # Use weights_only=True if available to avoid pickle warning
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    # Try to infer num_classes
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint if isinstance(checkpoint, dict) else {}
    # Try to get label map
    label_to_idx = {}
    if label_map_path and os.path.isfile(label_map_path):
        with open(label_map_path, 'r', encoding='utf-8') as f:
            label_to_idx = json.load(f)
    elif 'label_map' in checkpoint:
        label_to_idx = checkpoint['label_map']
    elif 'label_to_idx' in checkpoint:
        label_to_idx = checkpoint['label_to_idx']
    idx_to_label = {v: k for k, v in label_to_idx.items()} if label_to_idx else {}
    # Infer num_classes
    num_classes = None
    if 'classifier.1.weight' in state_dict:
        num_classes = state_dict['classifier.1.weight'].shape[0]
    elif 'fc.weight' in state_dict:
        num_classes = state_dict['fc.weight'].shape[0]
    elif label_to_idx:
        num_classes = len(label_to_idx)
    else:
        num_classes = 5  # fallback
    # Build model
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model, idx_to_label

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='dataset_clean/val', help='Folder z obrazami (ImageFolder)')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    model_path, label_map_path = find_latest_run_checkpoint()
    if not model_path:
        model_path = 'model/car_model2.pth'
        label_map_path = None
    print(f"Używam modelu: {model_path}")
    model, idx_to_label = load_model(model_path, label_map_path, device=args.device)

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    ds = datasets.ImageFolder(args.data, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False)

    # overall metrics
    correct = 0
    total = 0

    # per-class counters (indexed by class idx)
    per_totals = defaultdict(int)
    per_correct = defaultdict(int)

    for x, y in loader:
        x = x.to(args.device)
        y = y.to(args.device)
        with torch.no_grad():
            out = model(x)
            pred = out.argmax(1)
        correct += (pred==y).sum().item()
        total += x.size(0)
        # per-sample accumulation
        preds_cpu = pred.cpu().tolist()
        targets_cpu = y.cpu().tolist()
        for p, t in zip(preds_cpu, targets_cpu):
            per_totals[t] += 1
            if p == t:
                per_correct[t] += 1

    acc = correct/total if total else 0.0
    incorrect = total - correct
    err_pct = (incorrect/total)*100 if total else 0.0
    print(f"Dokładność na {args.data}: {acc:.4f} ({correct}/{total})")
    print(f"Błędnych predykcji: {incorrect} / {total} ({err_pct:.2f}%)")

    # Per-class percentages
    print("\nSzczegółowe wyniki per-klasa:")
    # Determine label names: prefer idx_to_label, fallback to dataset class names
    num_classes = max(max(per_totals.keys(), default=-1), max(per_correct.keys(), default=-1), len(ds.classes)-1) + 1
    for i in range(num_classes):
        total_i = per_totals.get(i, 0)
        correct_i = per_correct.get(i, 0)
        acc_i = (correct_i / total_i * 100.0) if total_i > 0 else 0.0
        if idx_to_label:
            label_name = idx_to_label.get(i, f"{i}")
        else:
            label_name = ds.classes[i] if i < len(ds.classes) else str(i)
        print(f"- {label_name}: {acc_i:.2f}% ({correct_i}/{total_i})")

    # Pokaż przykładowe predykcje
    print("\nPrzykładowe predykcje:")
    for i in range(len(ds)):
        img, label = ds[i]
        with torch.no_grad():
            out = model(img.unsqueeze(0).to(args.device))
            pred = out.argmax(1).item()
        label_name = ds.classes[label]
        pred_name = idx_to_label.get(pred, str(pred)) if idx_to_label else str(pred)
        print(f"{i+1}. Prawdziwa: {label_name}, Predykcja: {pred_name}")

if __name__ == '__main__':
    main()
