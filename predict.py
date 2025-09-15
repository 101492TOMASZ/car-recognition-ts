import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import traceback
import os
import json
from utils_paths import resolve_asset_path

# Reusable transforms (avoid recreating each call)
_CLASSIFY_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Simple cache to avoid reloading classifier repeatedly when caller passes None.
_CACHED_CLASSIFIER = None
_CACHED_LABELS = None
_CACHED_PTH = None

def find_latest_best_checkpoint(runs_dir=None):
    """Find the newest classifier checkpoint and its label map.

    Priority:
    1) In runs/<exp>/, prefer final.pth; fallback to best.pth. Choose newest mtime across runs.
    2) Fallback to top-level final.pth; then best.pth; label_map.json alongside if present.
    """
    # Prefer bundled runs/ inside the app when frozen; fall back to CWD 'runs'
    if runs_dir is None:
        try:
            candidate = str(resolve_asset_path('runs'))
            runs_dir = candidate if os.path.isdir(candidate) else 'runs'
        except Exception:
            runs_dir = 'runs'
    bests = []
    if os.path.isdir(runs_dir):
        try:
            for name in os.listdir(runs_dir):
                p = os.path.join(runs_dir, name)
                if not os.path.isdir(p):
                    continue
                cand_final = os.path.join(p, 'final.pth')
                cand_best = os.path.join(p, 'best.pth')
                cand = None
                if os.path.isfile(cand_final):
                    cand = cand_final
                elif os.path.isfile(cand_best):
                    cand = cand_best
                if cand is None:
                    continue
                lm = os.path.join(p, 'label_map.json')
                label_map = {}
                if os.path.isfile(lm):
                    try:
                        with open(lm, 'r') as f:
                            label_map = json.load(f)
                    except Exception:
                        label_map = {}
                try:
                    mtime = os.path.getmtime(cand)
                except Exception:
                    mtime = 0
                bests.append((mtime, cand, label_map))
        except Exception:
            pass
    if bests:
        bests.sort(reverse=True)
        return bests[0][1], bests[0][2]
    # Fallback to top-level files bundled next to the app
    for fname in ('final.pth', 'best.pth'):
        single_pth = str(resolve_asset_path(fname))
        if os.path.isfile(single_pth):
            single_map = str(resolve_asset_path('label_map.json'))
            label_map = {}
            if os.path.isfile(single_map):
                try:
                    with open(single_map, 'r') as f:
                        label_map = json.load(f)
                except Exception:
                    label_map = {}
            return single_pth, label_map
    return None, {}
    bests = []
    for name in os.listdir(runs_dir):
        p = os.path.join(runs_dir, name)
        if not os.path.isdir(p):
            continue
        cand = os.path.join(p, 'final.pth')  # changed from best.pth to final.pth
        lm = os.path.join(p, 'label_map.json')
        if os.path.isfile(cand):
            mtime = os.path.getmtime(cand)
            label_map = {}
            if os.path.isfile(lm):
                try:
                    with open(lm, 'r') as f:
                        label_map = json.load(f)
                except Exception:
                    label_map = {}
            bests.append((mtime, cand, label_map))
    if not bests:
        return None, {}
    bests.sort(reverse=True)
    return bests[0][1], bests[0][2]

def load_classifier(model_path, label_map, device='cpu'):
    import torch.nn as nn
    from torchvision import models
    num_classes = None
    idx_to_label = {}
    if isinstance(label_map, dict):
        try:
            idx_map = {int(k): v for k, v in label_map.items()}
            if idx_map:
                num_classes = max(idx_map.keys()) + 1
                idx_to_label = idx_map
        except Exception:
            try:
                inv_map = {int(v): k for k, v in label_map.items()}
                if inv_map:
                    num_classes = max(inv_map.keys()) + 1
                    idx_to_label = inv_map
            except Exception:
                idx_to_label = {}
                num_classes = 100
    elif isinstance(label_map, list):
        idx_to_label = {i: v for i, v in enumerate(label_map)}
        num_classes = len(label_map)
    if num_classes is None:
        num_classes = 100
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    try:
        state = torch.load(model_path, map_location=device)
        # Obsługa formatu {'model_state': ...} (z final.pth)
        if isinstance(state, dict):
            if 'model_state' in state:
                sd = state['model_state']
            elif 'state_dict' in state:
                sd = state['state_dict']
            else:
                sd = state
        else:
            sd = state
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            new_sd = {}
            for k, v in sd.items():
                nk = k.replace('module.', '')
                if nk.startswith('model.'):
                    nk = nk[len('model.'):]
                new_sd[nk] = v
            model.load_state_dict(new_sd, strict=False)
    except Exception as e:
        print(f"[load_classifier] ERROR loading model: {e}\n{traceback.format_exc()}")
    model.to(device)
    model.eval()
    return model, idx_to_label

def predict_image(
    image_path,
    yolo_model,
    classifier=None,
    idx_to_label=None,
    device='cpu',
    require_vehicle=True,
    min_conf=0.25,
    add_margin=0.0,
    verbose=True,
    logger=None,
):
    """Predict brand for a single image.

    Steps:
      1. (Optional) YOLO detection -> choose largest vehicle box (center-biased).
      2. Crop (optionally expand by ``add_margin`` fraction of box size).
      3. Single forward pass with gradients (for Grad-CAM) producing logits.
      4. Compute Grad-CAM heatmap & softmax probs from same pass (no duplicate pass).

    Returns dict: {brand, confidence, message, heatmap (base64 PNG or None), no_vehicle(bool)}
    """
    log_debug = (logger.debug if logger else print) if verbose else (lambda *a, **k: None)
    log_info = logger.info if logger else print
    log_warn = logger.warning if logger else print
    log_err = logger.error if logger else print
    if verbose:
        log_debug(f"[predict_image] Predicting for: {image_path}")

    # Lazy / cached classifier load
    global _CACHED_CLASSIFIER, _CACHED_LABELS, _CACHED_PTH
    if classifier is None or idx_to_label is None:
        best_pth, label_map = find_latest_best_checkpoint('runs')
        if best_pth is None:
            if verbose:
                log_warn("[predict_image] No classifier checkpoint found in runs/!")
            return {'brand': None, 'confidence': None, 'message': 'No classifier checkpoint found'}
        if _CACHED_CLASSIFIER is None or _CACHED_PTH != best_pth:
            if verbose:
                log_info(f"[predict_image] Loading classifier: {best_pth}")
            _CACHED_CLASSIFIER, _CACHED_LABELS = load_classifier(best_pth, label_map, device=device)
            _CACHED_PTH = best_pth
        classifier, idx_to_label = _CACHED_CLASSIFIER, _CACHED_LABELS
    heatmap_img = None
    img = Image.open(image_path).convert('RGB')
    crop_img = img
    crop_info = 'full image'
    biggest_box = None  # (area, [x1,y1,x2,y2])
    if yolo_model is not None:
        try:
            results = yolo_model(np.array(img))
        except Exception:
            results = yolo_model(img)
        xyxy = []
        conf = []
        cls = []
        try:
            r = results[0]
            boxes = getattr(r, 'boxes', None)
            if boxes is not None:
                xyxy = getattr(boxes, 'xyxy', None)
                conf = getattr(boxes, 'conf', None)
                cls = getattr(boxes, 'cls', None)
                if xyxy is not None:
                    try:
                        xyxy = xyxy.cpu().numpy()
                    except Exception:
                        xyxy = np.array(xyxy)
                if conf is not None:
                    try:
                        conf = conf.cpu().numpy()
                    except Exception:
                        conf = np.array(conf)
                if cls is not None:
                    try:
                        cls = cls.cpu().numpy()
                    except Exception:
                        cls = np.array(cls)
        except Exception:
            xyxy, conf, cls = [], [], []
        if verbose:
            log_debug(f"[predict_image] YOLO found {len(xyxy)} boxes.")
        # Vehicle gating & crop selection
        try:
            names = None
            if 'r' in locals() and hasattr(r, 'names'):
                names = getattr(r, 'names')
            elif hasattr(yolo_model, 'model') and hasattr(yolo_model.model, 'names'):
                names = getattr(yolo_model.model, 'names')
        except Exception:
            names = None
        allowed = {'car','samoch','auto','vehicle','automobile','truck','bus','van','pickup','suv','train'}
        img_area = max(1, img.width * img.height)
        candidates = []  # list of (area, box)
        max_area = 0.0
        for i, box in enumerate(xyxy if len(xyxy) else []):
            try:
                b = list(map(float, box))
            except Exception:
                continue
            area = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
            ci = None
            lbl = ''
            try:
                ci = int(cls[i]) if cls is not None and len(cls) > i else None
                if names is not None:
                    if isinstance(names, dict) and ci in names:
                        lbl = str(names[ci])
                    elif hasattr(names, '__len__') and ci is not None and ci < len(names):
                        lbl = str(names[ci])
            except Exception:
                lbl = ''
            cval = float(conf[i]) if conf is not None and len(conf) > i else 0.0
            lname = lbl.lower()
            is_vehicle = any(tok in lname for tok in allowed)
            if is_vehicle and cval >= float(min_conf) and area >= 0.01 * img_area:
                candidates.append((area, b))
            if area > max_area:
                biggest_box = (area, b)
                max_area = area
        if require_vehicle and not candidates:
            msg = 'Zdjęcie nie przedstawia pojazdu'
            if verbose:
                log_info(f"[predict_image] {msg} — aborting classification.")
            return {'brand': None, 'confidence': None, 'message': msg, 'heatmap': None, 'no_vehicle': True}

        # Select crop: prefer largest VEHICLE box; tie-break by closeness to image center.
        chosen = None
        if candidates:
            img_cx, img_cy = img.width / 2.0, img.height / 2.0
            ranked = []  # (-area, distance_sq, box)
            for area, b in candidates:
                try:
                    cx = (b[0] + b[2]) / 2.0
                    cy = (b[1] + b[3]) / 2.0
                    dist2 = (cx - img_cx)**2 + (cy - img_cy)**2
                except Exception:
                    dist2 = 1e12
                ranked.append((-area, dist2, b))
            ranked.sort()
            chosen = ranked[0][2]
        elif biggest_box:
            # fallback if no valid vehicle candidate but boxes exist (should be rare when require_vehicle False)
            chosen = biggest_box[1]

        if chosen is not None:
            x1, y1, x2, y2 = map(int, chosen)
            # Optional margin expansion
            if add_margin > 0:
                w_box = x2 - x1
                h_box = y2 - y1
                expand_w = int(w_box * add_margin)
                expand_h = int(h_box * add_margin)
                x1 -= expand_w
                y1 -= expand_h
                x2 += expand_w
                y2 += expand_h
            # Clamp to image bounds (PIL expects right/lower <= width/height)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.width, x2)
            y2 = min(img.height, y2)
            if x2 > x1 and y2 > y1:
                crop_img = img.crop((x1, y1, x2, y2))
                crop_info = f'crop_centered_largest_vehicle: ({x1},{y1},{x2},{y2})'

    # --- GRAD-CAM DLA MOBILENETV2 ---
    try:
        import cv2, io, base64
        # Single forward WITH grad to enable Grad-CAM + classification.
        image_tensor = _CLASSIFY_TRANSFORM(crop_img).unsqueeze(0).to(device)
        image_tensor.requires_grad_(True)
        classifier.eval()

        fmap = None
        grad = None
        target_layer = classifier.features[-1]

        def forward_hook(_m, _i, o):
            nonlocal fmap
            fmap = o.detach()

        def backward_full_hook(_m, gin, gout):
            # gin unused; gout is tuple
            nonlocal grad
            grad = gout[0].detach()

        h1 = target_layer.register_forward_hook(forward_hook)
        # use full backward hook for future compatibility
        h2 = target_layer.register_full_backward_hook(backward_full_hook)

        out = classifier(image_tensor)
        pred_class = out.argmax(dim=1).item()
        score = out[0, pred_class]
        classifier.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)
        h1.remove(); h2.remove()

        # Softmax probabilities (detach to avoid further graph use)
        probs = torch.softmax(out.detach(), dim=1)
        conf_val, pred = torch.max(probs, 1)
        predicted_idx = int(pred.item())
        predicted_label = idx_to_label.get(predicted_idx, str(predicted_idx))
        confidence = float(conf_val.item() * 100.0)

        # Grad-CAM build (only if hooks succeeded)
        if fmap is not None and grad is not None:
            try:
                weights = grad.mean(dim=[2, 3], keepdim=True)
                cam = (weights * fmap).sum(dim=1, keepdim=True)
                cam = cam.squeeze().cpu().numpy()
                cam = np.maximum(cam, 0)
                cam = (cam - cam.min()) / (np.ptp(cam) + 1e-8)
                cam_img = (cam * 255).astype(np.uint8)
                cam_img = cv2.resize(cam_img, (crop_img.width, crop_img.height))
                crop_np = np.array(crop_img.convert('RGB'))
                heatmap_color = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(crop_np, 0.5, heatmap_color, 0.5, 0)
                heatmap_pil = Image.fromarray(overlay)
                buf = io.BytesIO()
                heatmap_pil.save(buf, format='PNG')
                heatmap_img = base64.b64encode(buf.getvalue()).decode('utf-8')
            except Exception as e:
                if verbose:
                    log_warn(f"[predict_image] Grad-CAM postprocess failed: {e}")
                heatmap_img = None
        else:
            heatmap_img = None

        if verbose:
            log_debug(f"[predict_image] Using {crop_info}")
            log_debug(f"[predict_image] Tensor shape: {image_tensor.shape}, min={image_tensor.min().item():.4f}, max={image_tensor.max().item():.4f}")
            log_debug(f"[predict_image] Raw logits: {out.detach().cpu().numpy()}")
            log_debug(f"[predict_image] Softmax: {probs.cpu().numpy()}")
            log_debug(f"[predict_image] Predicted idx: {predicted_idx}")
            log_debug(f"[predict_image] idx_to_label: {idx_to_label}")
            log_info(f"[predict_image] Result: brand={predicted_label}, confidence={confidence:.2f}%")

        return {
            'brand': predicted_label,
            'confidence': confidence,
            'message': None,
            'heatmap': heatmap_img,
            'no_vehicle': False
        }
    except Exception as e:
        tb = traceback.format_exc()
        if verbose:
            log_err(f"[predict_image] ERROR: {e}\n{tb}")
        return {'brand': None, 'confidence': None, 'message': f'{e}\n{tb}', 'heatmap': None, 'no_vehicle': False}
