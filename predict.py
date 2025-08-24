import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import traceback
import os
import json

def find_latest_best_checkpoint(runs_dir='runs'):
    if not os.path.isdir(runs_dir):
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

def predict_image(image_path, yolo_model, classifier=None, idx_to_label=None, device='cpu'):
    """
    Runs YOLO detection, crops the car, and classifies with MobileNetV2.
    If classifier or idx_to_label is None, loads the latest from runs/.
    Returns: dict with keys 'brand', 'confidence', 'message' (None if ok, else error msg)
    """
    print(f"[predict_image] Predicting for: {image_path}")
    if classifier is None or idx_to_label is None:
        best_pth, label_map = find_latest_best_checkpoint('runs')
        if best_pth is None:
            print("[predict_image] No classifier checkpoint found in runs/!")
            return {'brand': None, 'confidence': None, 'message': 'No classifier checkpoint found'}
        classifier, idx_to_label = load_classifier(best_pth, label_map, device=device)
        print(f"[predict_image] Loaded classifier: {best_pth}")
    heatmap_img = None
    img = Image.open(image_path).convert('RGB')
    crop_img = img
    crop_info = 'full image'
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
        print(f"[predict_image] YOLO found {len(xyxy)} boxes.")
        # Crop do największego bounding boxa niezależnie od klasy
        biggest_box = None
        max_area = 0
        for i, box in enumerate(xyxy if len(xyxy) else []):
            try:
                b = list(map(float, box))
            except Exception:
                continue
            area = (b[2] - b[0]) * (b[3] - b[1])
            print(f"  box {i}: area={area}, coords={b}")
            if area > max_area:
                biggest_box = b
                max_area = area
        if biggest_box is None and len(xyxy) == 0:
            crop_img = img
            crop_info = 'full image (no detection)'
        else:
            chosen = biggest_box
            if chosen is not None:
                x1, y1, x2, y2 = map(int, chosen)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.width - 1, x2), min(img.height - 1, y2)
                if x2 > x1 and y2 > y1:
                    crop_img = img.crop((x1, y1, x2, y2))
                    crop_info = f'crop: ({x1},{y1},{x2},{y2})'

    # --- GRAD-CAM DLA MOBILENETV2 ---
    try:
        import cv2
        import io, base64
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(crop_img).unsqueeze(0).to(device)
        image_tensor.requires_grad = True
        # Forward + backward na predykcję
        classifier.eval()
        fmap = None
        grad = None
        def forward_hook(module, input, output):
            nonlocal fmap
            fmap = output.detach()
        def backward_hook(module, grad_in, grad_out):
            nonlocal grad
            grad = grad_out[0].detach()
        handle_fwd = classifier.features[-1].register_forward_hook(forward_hook)
        handle_bwd = classifier.features[-1].register_backward_hook(backward_hook)
        out = classifier(image_tensor)
        pred_class = out.argmax(dim=1).item()
        score = out[0, pred_class]
        classifier.zero_grad()
        score.backward(retain_graph=True)
        handle_fwd.remove()
        handle_bwd.remove()
        # Grad-CAM: waga = średnia po spatial grad
        weights = grad.mean(dim=[2, 3], keepdim=True)  # shape (1, C, 1, 1)
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
        print(f"[predict_image] Grad-CAM generation failed: {e}")
        heatmap_img = None
    print(f"[predict_image] Using {crop_info}")
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(crop_img).unsqueeze(0).to(device)
        print(f"[predict_image] Tensor shape: {image_tensor.shape}, min={image_tensor.min().item():.4f}, max={image_tensor.max().item():.4f}")
        classifier.eval()
        with torch.no_grad():
            out = classifier(image_tensor)
            probs = torch.softmax(out, dim=1)
        print(f"[predict_image] Raw logits: {out.cpu().numpy()}")
        print(f"[predict_image] Softmax: {probs.cpu().numpy()}")
        conf_val, pred = torch.max(probs, 1)
        predicted_idx = int(pred.item())
        print(f"[predict_image] Predicted idx: {predicted_idx}")
        print(f"[predict_image] idx_to_label: {idx_to_label}")
        predicted_label = idx_to_label.get(predicted_idx, str(predicted_idx))
        confidence = float(conf_val.item() * 100.0)
        print(f"[predict_image] Result: brand={predicted_label}, confidence={confidence:.2f}%")
        return {'brand': predicted_label, 'confidence': confidence, 'message': None, 'heatmap': heatmap_img}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[predict_image] ERROR: {e}\n{tb}")
        return {'brand': None, 'confidence': None, 'message': f'{e}\n{tb}', 'heatmap': None}
