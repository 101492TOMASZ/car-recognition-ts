#!/usr/bin/env python3
"""Train a MobileNetV2 classifier on a dataset prepared with split_and_dedupe.py.

Usage examples:
  python training/train_mobile.py --data dataset_clean --epochs 10 --batch 32 --out runs/run1

Expect `--data` to contain `train/` and `val/` subfolders following torchvision ImageFolder layout.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to dataset root containing train/ and val/ folders")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--out", default="runs/run1", help="Output folder for checkpoints and label map")
    p.add_argument("--pretrained", action="store_true", help="Start from torchvision pretrained weights")
    p.add_argument("--freeze-epochs", type=int, default=2, help="Number of epochs to train only head (freeze backbone)")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to continue training')
    return p.parse_args()


def get_transforms(img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, val_tf


def build_dataloaders(data_root, img_size, batch, workers):
    data_root = Path(data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Expected train/ and val/ under {data_root}")

    train_tf, val_tf = get_transforms(img_size)
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, train_ds.classes


def build_model(num_classes, pretrained=True, device="cpu"):
    model = models.mobilenet_v2(pretrained=pretrained)
    # Replace classifier (Dropout, Linear)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(device)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    losses = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            losses += float(loss.item()) * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += imgs.size(0)
    return losses / total, correct / total if total > 0 else 0.0


def evaluate_per_class(model, loader, device, classes):
    """
    Return per-class counts and percentages:
      { class_name: {"correct": int, "total": int, "accuracy": float_percent} }
    """
    model.eval()
    n_classes = len(classes)
    correct_counts = [0] * n_classes
    total_counts = [0] * n_classes
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().tolist()
            tg = targets.cpu().tolist()
            for p, t in zip(preds, tg):
                total_counts[t] += 1
                if p == t:
                    correct_counts[t] += 1
    results = {}
    for i, cname in enumerate(classes):
        tot = total_counts[i]
        corr = correct_counts[i]
        acc = (corr / tot * 100.0) if tot > 0 else 0.0
        results[cname] = {"correct": int(corr), "total": int(tot), "accuracy_pct": round(acc, 2)}
    return results


def save_checkpoint(state, out_dir, name="checkpoint.pth"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    torch.save(state, path)
    return path


def train(args):
    device = args.device
    train_loader, val_loader, classes = build_dataloaders(args.data, args.img_size, args.batch, args.workers)
    num_classes = len(classes)
    print(f"Found {num_classes} classes: {classes}")
    model = build_model(num_classes, pretrained=args.pretrained, device=device)

    # Save label map
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({c: i for i, c in enumerate(classes)}, f, ensure_ascii=False, indent=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    start_epoch = 0
    best_val_acc = 0.0

    # --- START: domyślne wznawianie z final.pth ---
    if not args.checkpoint:
        final_ckpt = out_dir / "final.pth"
        if final_ckpt.exists():
            args.checkpoint = str(final_ckpt)
            print(f"Auto-resume: found final.pth, resuming from {args.checkpoint}")
    # --- END ---

    # Load existing checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=False)
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)

    scaler = torch.cuda.amp.GradScaler() if device.startswith("cuda") else None

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()

        # Freeze backbone for first N epochs
        if epoch < args.freeze_epochs:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True

        running_loss = 0.0
        running_samples = 0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item()) * imgs.size(0)
            running_samples += imgs.size(0)

        scheduler.step()
        train_loss = running_loss / running_samples if running_samples else 0.0
        val_loss, val_acc = evaluate(model, val_loader, device)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{args.epochs}  time={epoch_time:.1f}s  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # Save checkpoint
        ck = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "args": vars(args),
        }
        save_checkpoint(ck, str(out_dir), name=f"checkpoint_epoch_{epoch+1}.pth")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(ck, str(out_dir), name="best.pth")

    # Final save
    save_checkpoint({"model_state": model.state_dict(), "args": vars(args)}, str(out_dir), name="final.pth")
    # compute and save per-class results on validation set
    print("Computing per-class validation results...")
    per_class = evaluate_per_class(model, val_loader, device, classes)
    per_class_path = out_dir / "per_class_results.json"
    with open(per_class_path, "w", encoding="utf-8") as f:
        json.dump(per_class, f, ensure_ascii=False, indent=2)
    print(f"Per-class validation results saved to: {per_class_path}")
    print(f"Training finished. Best val acc: {best_val_acc:.4f}. Artifacts saved to {out_dir}")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
