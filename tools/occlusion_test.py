import argparse
import tempfile
import os
from PIL import Image, ImageDraw
import numpy as np
import cv2
from predict import predict_image

def occlusion_test(image_path, step=40, window=80, device='cpu', out_path='occlusion_map.png'):
    img = Image.open(image_path).convert('RGB')
    w, h = img.size

    # baseline prediction
    baseline = predict_image(image_path, None, None, None, device=device)
    base_brand = baseline.get('brand')
    base_conf = baseline.get('confidence', 0.0)
    if base_brand is None:
        raise RuntimeError(f"Baseline prediction failed: {baseline.get('message')}")

    # heat accumulator (sum of drops) and counts for averaging
    heat = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    print(f"Baseline: {base_brand}, confidence={base_conf:.2f}%")
    tmp_files = []
    try:
        for y in range(0, h, step):
            for x in range(0, w, step):
                # build masked image (black rectangle)
                im_mask = img.copy()
                draw = ImageDraw.Draw(im_mask)
                x1 = x
                y1 = y
                x2 = min(x + window, w)
                y2 = min(y + window, h)
                draw.rectangle([x1, y1, x2, y2], fill=(0,0,0))
                # save to temp file
                tf = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                tmp_path = tf.name
                tf.close()
                im_mask.save(tmp_path, quality=90)
                tmp_files.append(tmp_path)

                # predict masked image
                res = predict_image(tmp_path, None, None, None, device=device)
                pred_brand = res.get('brand')
                pred_conf = res.get('confidence', 0.0)

                # measure remaining confidence for baseline brand:
                # if predict_image does not return per-class probs, approximate:
                # if predicted brand == baseline -> keep its confidence, else set 0
                remaining = pred_conf if pred_brand == base_brand else 0.0
                drop = max(0.0, base_conf - remaining)

                # accumulate drop into heat map (add to masked rectangle)
                heat[y1:y2, x1:x2] += drop
                counts[y1:y2, x1:x2] += 1.0

        # normalize by counts to get average drop per-pixel
        mask = counts > 0
        heat[mask] = heat[mask] / counts[mask]
        # smooth heatmap, normalize 0..1
        heat = cv2.GaussianBlur(heat, (0,0), sigmaX=window/2)
        if heat.max() > 0:
            heat_norm = (heat - heat.min()) / (heat.max() - heat.min())
        else:
            heat_norm = heat

        # create overlay on original image
        orig = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        heat_col = (heat_norm * 255).astype(np.uint8)
        heat_col = cv2.applyColorMap(heat_col, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig, 0.6, heat_col, 0.4, 0)

        # save combined image
        cv2.imwrite(out_path, overlay)
        print(f"Occlusion map saved to {out_path}")
    finally:
        for p in tmp_files:
            try:
                os.remove(p)
            except Exception:
                pass

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='Input image path')
    ap.add_argument('--step', type=int, default=40, help='Grid step (px)')
    ap.add_argument('--window', type=int, default=80, help='Occluder window size (px)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='occlusion_map.png')
    args = ap.parse_args()
    occlusion_test(args.image, args.step, args.window, device=args.device, out_path=args.out)