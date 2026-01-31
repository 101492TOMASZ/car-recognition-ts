"""
Split and deduplicate dataset.
- Input: directory with class subfolders (dataset/{CLASS}/img.jpg)
- Output: output_dir/{train,val,test}/{CLASS}/img.jpg
- Removes near-duplicates using perceptual hashing (imagehash)
- Writes metadata CSV with original path, split, class, hash

Usage:
    python training/split_and_dedupe.py --input dataset_german --output dataset_clean --val 0.1 --test 0.1
"""
import os
import argparse
import csv
import shutil
from pathlib import Path
from collections import defaultdict
import random

from PIL import Image
import imagehash


def collect_images(input_dir):
    input_dir = Path(input_dir)
    items = []
    for cls in sorted([p.name for p in input_dir.iterdir() if p.is_dir()]):
        cls_dir = input_dir / cls
        for p in cls_dir.iterdir():
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']:
                items.append((str(p), cls))
    return items


def dedupe_items(items, hashfunc='phash', threshold=5):
    """Return list of kept items after removing perceptual duplicates.
    We keep first seen image among similar ones (hamming distance <= threshold).
    """
    kept = []
    hashes = []
    for path, cls in items:
        try:
            h = imagehash.phash(Image.open(path))
        except Exception:
            continue
        dropped = False
        for prev_h in hashes:
            if h - prev_h <= threshold:
                dropped = True
                break
        if not dropped:
            kept.append((path, cls, str(h)))
            hashes.append(h)
    return kept


def split_and_copy(kept, output_dir, val_frac=0.1, test_frac=0.1, seed=42):
    random.seed(seed)
    by_class = defaultdict(list)
    for path, cls, h in kept:
        by_class[cls].append((path, h))

    out = Path(output_dir)
    for split in ['train', 'val', 'test']:
        for cls in by_class.keys():
            (out / split / cls).mkdir(parents=True, exist_ok=True)

    rows = []
    for cls, files in by_class.items():
        random.shuffle(files)
        n = len(files)
        n_val = int(n * val_frac)
        n_test = int(n * test_frac)
        n_train = n - n_val - n_test
        idx = 0
        for i, (path, h) in enumerate(files):
            if i < n_train:
                split = 'train'
            elif i < n_train + n_val:
                split = 'val'
            else:
                split = 'test'
            dst = out / split / cls / Path(path).name
            try:
                shutil.copy2(path, dst)
            except Exception:
                continue
            rows.append({'original': path, 'split': split, 'class': cls, 'hash': h, 'dest': str(dst)})
    # write metadata
    with open(out / 'metadata.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['original', 'dest', 'split', 'class', 'hash'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--val', type=float, default=0.1)
    parser.add_argument('--test', type=float, default=0.1)
    parser.add_argument('--threshold', type=int, default=5, help='hamming threshold for phash')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    items = collect_images(args.input)
    print(f"Found {len(items)} images")
    kept = dedupe_items(items, threshold=args.threshold)
    print(f"Kept {len(kept)} images after deduplication")
    out = split_and_copy(kept, args.output, val_frac=args.val, test_frac=args.test, seed=args.seed)
    print(f"Created dataset at {out}")

if __name__ == '__main__':
    main()
