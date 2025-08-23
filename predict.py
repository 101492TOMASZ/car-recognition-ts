import time
import logging
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from utils.logger import get_logger

log = get_logger('predict')
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# === KONFIGURACJA ===
def resource_path(relative_path):
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return os.path.join(getattr(sys, '_MEIPASS'), relative_path)
    return os.path.join(os.path.abspath('.'), relative_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Determine model path: prefer latest run under `runs/` (best.pth -> final.pth -> any checkpoint),
# otherwise fallback to packaged model `model/car_model2.pth`.

# Always use the latest best.pth from runs/ as the model for GUI and CLI
def find_latest_best_checkpoint(runs_dir='runs'):
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
        if os.path.isfile(best):
            return best, os.path.join(run, 'label_map.json')
    return None, None

MODEL_PATH, LABEL_MAP_PATH = find_latest_best_checkpoint('runs')
if MODEL_PATH is None:
    MODEL_PATH = resource_path('model/car_model2.pth')  # legacy fallback
    LABEL_MAP_PATH = None
IMAGE_PATH = "cropped_car.jpg"  # path used for debugging if needed

# device and classifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load detection model (YOLOv8n) if available
if YOLO is not None:
    try:
        yolo = YOLO('yolov8n.pt')
    except Exception:
        # fall back to model name which will download weights
        yolo = YOLO('yolov8n')
else:
    yolo = None

# Load checkpoint (prefer weights_only to avoid pickle risks when supported)
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
except TypeError:
    # older torch versions may not support weights_only argument
    checkpoint = torch.load(MODEL_PATH, map_location=device)
except FileNotFoundError:
    # if MODEL_PATH missing, raise a clearer error
    raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")

# Extract state_dict and label maps if present
if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
    state_dict = checkpoint['model_state']
    # prefer label_map.json saved alongside run
    label_to_idx = checkpoint.get('label_map', {}) or checkpoint.get('label_to_idx', {}) or {}
    idx_to_label = {v: k for k, v in label_to_idx.items()} if label_to_idx else {}
elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    label_to_idx = checkpoint.get('label_to_idx', {}) or {}
    idx_to_label = {v: k for k, v in label_to_idx.items()} if label_to_idx else {}
else:
    # assume checkpoint is a raw state_dict or mapping
    state_dict = checkpoint if isinstance(checkpoint, dict) else {}
    label_to_idx = {}
    idx_to_label = {}

# If we didn't get a label_map from the checkpoint, try loading label_map.json from the run folder
if (not label_to_idx or not idx_to_label) and LABEL_MAP_PATH and os.path.isfile(LABEL_MAP_PATH):
    try:
        import json
        with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
            label_to_idx = json.load(f)
            idx_to_label = {v: k for k, v in label_to_idx.items()}
    except Exception:
        log.exception('Failed to load label_map.json at %s', LABEL_MAP_PATH)

# Inspect keys to decide which architecture to instantiate
state_keys = list(state_dict.keys()) if isinstance(state_dict, dict) else []
use_resnet = any(k.startswith('layer') or k.startswith('conv1') or k.startswith('fc.') or 'fc.weight' in k for k in state_keys)
use_mobilenet = any(k.startswith('features') or k.startswith('classifier') for k in state_keys)

# try to infer num_classes from checkpoint
inferred_num_classes = None
if any(k == 'fc.weight' for k in state_keys):
    try:
        inferred_num_classes = state_dict['fc.weight'].shape[0]
    except Exception:
        inferred_num_classes = None
if any(k == 'classifier.1.weight' for k in state_keys):
    try:
        inferred_num_classes = state_dict['classifier.1.weight'].shape[0]
    except Exception:
        pass
if not inferred_num_classes and label_to_idx:
    inferred_num_classes = len(label_to_idx)

if use_resnet and not use_mobilenet:
    # instantiate ResNet50 to load legacy checkpoints
    model = models.resnet50(weights=None)
    # if we inferred num_classes, replace fc head accordingly
    if inferred_num_classes:
        model.fc = nn.Linear(model.fc.in_features, inferred_num_classes)
    model_loaded = True
elif use_mobilenet and not use_resnet:
    model = models.mobilenet_v2(weights=None)
    # ensure classifier head exists; classifier[1] is Linear
    if inferred_num_classes:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, inferred_num_classes)
    model_loaded = True
else:
    # default to MobileNetV2 (modern choice), loading may fail if checkpoint is incompatible
    model = models.mobilenet_v2(weights=None)
    model_loaded = True

# Attempt to load state_dict into model; allow strict=False to be flexible
try:
    model.load_state_dict(state_dict, strict=False)
except Exception as e:
    # try loading nested dict
    try:
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'], strict=False)
        else:
            raise
    except Exception as e2:
        raise RuntimeError(f"Unable to load model checkpoint: {e} / {e2}")

model = model.to(device)
model.eval()
log.info("Loaded classifier from %s on device %s", MODEL_PATH, device)

# Dynamic mapping
brand_to_id = {label: idx for label, idx in label_to_idx.items()} if label_to_idx else {}



# === FUNKCJA DO GRAD-CAM ===
def generate_gradcam(image_tensor, model, target_class, upsample_size=(224, 224)):
    """
    Compute Grad-CAM and return a normalized float32 numpy array in 0..1
    sized to `upsample_size`.
    """
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    # Find last Conv2d
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is None:
        raise RuntimeError("No Conv2d layer found in the model for Grad-CAM")

    # register hooks
    handle_fwd = last_conv.register_forward_hook(forward_hook)
    # register_backward_hook is deprecated but still works; fallback to full hook if available
    try:
        handle_bwd = last_conv.register_full_backward_hook(backward_hook)
    except Exception:
        handle_bwd = last_conv.register_backward_hook(backward_hook)

    # Forward + backward
    output = model(image_tensor)
    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()

    act = activations[0]   # [1, C, H, W]
    grad = gradients[0]    # [1, C, H, W]
    weights = grad.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
    gradcam_map = (weights * act).sum(dim=1, keepdim=True)  # [1, 1, H, W]
    gradcam_map = torch.relu(gradcam_map).squeeze().cpu().numpy()

    # Normalize to 0..1
    gradcam_map = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min() + 1e-8)

    # Resize to desired upsample size using PIL
    gradcam_uint8 = np.uint8(255 * gradcam_map)
    gradcam_img = Image.fromarray(gradcam_uint8).resize(upsample_size, resample=Image.BILINEAR)
    gradcam_resized = np.array(gradcam_img).astype(np.float32) / 255.0

    handle_fwd.remove()
    handle_bwd.remove()

    return gradcam_resized


# === FUNKCJA DO PREDYKCJI ===
def predict_image(image):
    # Transformacje obrazu (zgodne z tune_model.py)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # classifier input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    t0 = time.perf_counter()
    # Keep a copy of the original image and size
    orig_image = image.convert('RGB')
    orig_size = orig_image.size  # (width, height)

    # First: run YOLO to detect car and crop. If YOLO is not available or doesn't detect, use full image
    crop_image = image
    try:
        if yolo is not None:
            # ultralytics expects numpy array (H,W,3) RGB or BGR; pass RGB
            img_np = np.array(image.convert('RGB'))
            results = yolo(img_np, imgsz=640, conf=0.25, verbose=False)
            # results is a list-like; take first
            if len(results) > 0:
                r = results[0]
                boxes = getattr(r, 'boxes', None)
                if boxes is not None and len(boxes) > 0:
                    # boxes.xyxy, boxes.cls, boxes.conf
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else None
                    confs = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else None
                    # prefer detections with COCO car class (2)
                    chosen = None
                    for i, x in enumerate(xyxy):
                        c = int(cls[i]) if cls is not None else -1
                        if c == 2 or chosen is None:
                            # choose highest confidence among cars or fallback to first box
                            if chosen is None:
                                chosen = (x, confs[i] if confs is not None else 1.0, c)
                            else:
                                if confs is not None and confs[i] > chosen[1]:
                                    chosen = (x, confs[i], c)
                    if chosen is not None:
                        x1, y1, x2, y2 = map(int, chosen[0])
                        # clip
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(image.width - 1, x2), min(image.height - 1, y2)
                        if x2 > x1 and y2 > y1:
                            crop_image = image.crop((x1, y1, x2, y2))
    except Exception:
        log.exception('YOLO detection failed, continuing with full image')

    # Prepare classifier input
    image_tensor = transform(crop_image).unsqueeze(0).to(device)

    # Forward to get probabilities (no grad needed)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)

    confidence, predicted = torch.max(probabilities, 1)
    predicted_idx = predicted.item()
    predicted_class = idx_to_label.get(predicted_idx, str(predicted_idx))
    predicted_id = brand_to_id.get(predicted_class, "Nieznane ID")
    confidence_percent = confidence.item() * 100

    log.info("Model przewiduje: %s (id=%s) pewność=%.2f%%", predicted_class, predicted_id, confidence_percent)

    # Compute Grad-CAM on the cropped image for visualization
    try:
        image_tensor_for_cam = image_tensor.clone().detach().requires_grad_(True)
        gradcam_map = generate_gradcam(image_tensor_for_cam, model, predicted_idx)

        # Apply a colormap to the Grad-CAM map and resize to original image size
        cmap = plt.get_cmap('jet')
        heatmap_rgb = cmap(gradcam_map)[:, :, :3]  # HxWx3 float
        heatmap_uint8 = np.uint8(heatmap_rgb * 255)
        heatmap_img = Image.fromarray(heatmap_uint8).resize(crop_image.size, resample=Image.BILINEAR)
        blended = Image.blend(crop_image.convert('RGB'), heatmap_img.convert('RGB'), alpha=0.5)
    except Exception:
        log.exception('Grad-CAM failed')
        blended = crop_image

    t1 = time.perf_counter()
    log.info("Prediction finished in %.1f ms", (t1 - t0) * 1000)
    return {
        "brand": predicted_class,
        "confidence": confidence_percent,
        "heatmap": blended
    }