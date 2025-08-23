"""
Downloader for 5 German car brands with basic quality filters.
- Brands: BMW, Audi, Mercedes, Volkswagen, Porsche
- Uses BingImageCrawler to fetch images for several queries per brand
- Validates images: opens with PIL, minimum resolution, optional YOLO car detection
- Removes corrupt and too-small images

Usage:
    python training/downloader_german.py --output dataset_german --per-brand 500 --use-yolo

"""
import os
import argparse
from icrawler.builtin import BingImageCrawler
from PIL import Image, ImageFilter
from tqdm import tqdm
import numpy as np
import random

# optional YOLO check
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except Exception:
    YOLO = None
    _YOLO_AVAILABLE = False

# Basic configuration
GERMAN_BRANDS = ["BMW", "Audi", "Mercedes", "Volkswagen", "Porsche"]
TAGS = ["car", "vehicle", "automobile", "sedan", "SUV", "hatchback", "coupe", "convertible"]
# view-specific tags to capture multiple viewpoints (front, rear, side, interior, etc.)
VIEW_TAGS = [
    "front view", "rear view", "side view", "three quarter view", "rear", "front", "left side", "right side", "interior", "dashboard", "back"
]
YEARS = ["2010s", "2020s", "2000s"]
MIN_WIDTH = 200
MIN_HEIGHT = 200


def validate_image(path, min_w=MIN_WIDTH, min_h=MIN_HEIGHT, yolo_model=None):
    """Open image, check size, and optionally run YOLO detection for 'car' class.
    Returns True if image passes checks, False otherwise."""
    try:
        with Image.open(path) as im:
            im = im.convert('RGB')
            w, h = im.size
            if w < min_w or h < min_h:
                return False
            if yolo_model is not None:
                try:
                    # ultralytics expects numpy array
                    arr = np.array(im)
                    res = yolo_model(arr, imgsz=640, conf=0.25, verbose=False)
                    if len(res) and hasattr(res[0], 'boxes') and len(res[0].boxes) > 0:
                        # check for class id that corresponds to car (COCO: 2 is 'car')
                        try:
                            cls = res[0].boxes.cls.cpu().numpy()
                            # if any detected class equals 2 (car) or 3 (motorcycle) consider pass
                            if any((int(c) == 2 or int(c) == 3) for c in cls):
                                return True
                            else:
                                return False
                        except Exception:
                            # if parsing fails, accept the image (don't be too strict)
                            return True
                    else:
                        return False
                except Exception:
                    # YOLO run failed for this image; fallback to size pass
                    return True
            return True
    except Exception:
        return False


def download_brand(brand, output_dir, per_tag=200, yolo_model=None):
    brand_dir = os.path.join(output_dir, brand.upper())
    os.makedirs(brand_dir, exist_ok=True)
    existing = len(os.listdir(brand_dir))

    # Build queries mixing brand + tags + year ranges and views to improve variety
    queries = []
    for tag in TAGS:
        queries.append(f"{brand} {tag}")
    for view in VIEW_TAGS:
        queries.append(f"{brand} {view}")
    for year in YEARS:
        queries.append(f"{brand} {year}")
    # Add a few extra generic queries
    queries += [f"{brand} full view", f"{brand} 3/4 view", f"{brand} profile view"]

    # Crawl images
    crawler = BingImageCrawler(storage={'root_dir': brand_dir})
    for q in queries:
        # file_idx_offset ensures we don't overwrite existing files in the brand folder
        try:
            crawler.crawl(keyword=q, max_num=per_tag, file_idx_offset=existing)
        except Exception as e:
            print(f"Crawler error for query '{q}': {e}")
        existing += per_tag

    # Post-process: validate and remove bad images
    files = list(os.listdir(brand_dir))
    kept = 0
    kept_files = []
    for fname in tqdm(files, desc=f"Validating {brand}"):
        path = os.path.join(brand_dir, fname)
        ok = validate_image(path, yolo_model=yolo_model)
        if not ok:
            try:
                os.remove(path)
            except Exception:
                pass
        else:
            kept += 1
            kept_files.append(path)
    print(f"{brand}: kept {kept} images after validation")

    return kept_files


def main(args):
    output_dir = args.output
    per_brand = args.per_brand
    use_yolo = args.use_yolo and _YOLO_AVAILABLE

    if args.use_yolo and not _YOLO_AVAILABLE:
        print("ultralytics (YOLO) not available in the environment; continuing without detection filter.")

    yolo_model = None
    if use_yolo:
        # load yolov8n weights (this will download if missing)
        try:
            yolo_model = YOLO('yolov8n.pt')
        except Exception:
            yolo_model = YOLO('yolov8n')

    os.makedirs(output_dir, exist_ok=True)

    # Distribute desired count across queries: keep per_tag such that total >= per_brand
    queries_per_brand = len(TAGS) + len(YEARS) + 3
    per_tag = max(50, per_brand // queries_per_brand + 1)

    for brand in GERMAN_BRANDS:
        print(f"Downloading images for {brand} into {output_dir}/{brand.upper()}")
        kept = download_brand(brand, output_dir, per_tag=per_tag, yolo_model=yolo_model)

        # Optionally augment some of the kept images to increase robustness
        if args.augment:
            aug_prob = args.aug_prob
            aug_count = args.aug_count
            print(f"Augmenting images for {brand}: prob={aug_prob}, count={aug_count}")
            for path in kept:
                try:
                    if random.random() < aug_prob:
                        img = Image.open(path).convert('RGB')
                        base_name = os.path.splitext(os.path.basename(path))[0]
                        for i in range(aug_count):
                            aug_img = _simple_augment(img)
                            aug_name = f"aug_{base_name}_v{i+1}.jpg"
                            aug_path = os.path.join(output_dir, brand.upper(), aug_name)
                            aug_img.save(aug_path)
                except Exception:
                    continue

    print("Download finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download German car brand images for training')
    parser.add_argument('--output', '-o', type=str, default='dataset_german', help='output dataset directory')
    parser.add_argument('--per-brand', '-n', type=int, default=500, help='target images per brand (approx)')
    parser.add_argument('--use-yolo', action='store_true', help='use YOLO to validate that images contain cars (requires ultralytics)')
    parser.add_argument('--augment', action='store_true', help='create augmented variants of some downloaded images')
    parser.add_argument('--aug-prob', type=float, default=0.3, help='probability for each kept image to be augmented (0..1)')
    parser.add_argument('--aug-count', type=int, default=1, help='number of augmented variants to create per selected image')
    args = parser.parse_args()
    main(args)


# lightweight local augment used when --augment is enabled
def _simple_augment(img):
    # img is PIL RGB
    out = img.resize((224, 224))
    # brightness/contrast
    if random.random() < 0.6:
        out = Image.fromarray(np.uint8(np.clip(np.array(out) * random.uniform(0.6, 1.4), 0, 255)))
    # small rotations
    if random.random() < 0.2:
        out = out.rotate(random.uniform(-15, 15))
    # gaussian blur
    if random.random() < 0.2:
        out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2)))
    # occlusion
    if random.random() < 0.15:
        w, h = out.size
        rw = int(w * random.uniform(0.05, 0.2))
        rh = int(h * random.uniform(0.05, 0.2))
        x = random.randint(0, w - rw)
        y = random.randint(0, h - rh)
        draw = Image.new('RGB', (rw, rh), (random.randint(0, 80), random.randint(0, 80), random.randint(0, 80)))
        out.paste(draw, (x, y))
    # JPEG compression
    from io import BytesIO
    if random.random() < 0.2:
        buf = BytesIO()
        q = random.randint(30, 85)
        out.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        out = Image.open(buf).convert('RGB')
    return out
