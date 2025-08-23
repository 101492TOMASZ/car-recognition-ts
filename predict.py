import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# === KONFIGURACJA ===
def resource_path(relative_path):
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return os.path.join(getattr(sys, '_MEIPASS'), relative_path)
    return os.path.join(os.path.abspath('.'), relative_path)

MODEL_PATH = resource_path('model/car_model2.pth')  # Always use this for torch.load
IMAGE_PATH = "cropped_car.jpg"  # Ścieżka do obrazu do predykcji

# Wczytaj model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)  # ResNet50 bez wstępnych wag
model.fc = nn.Linear(model.fc.in_features, 10)  # Liczba klas (dopasowana do tune_model.py)

# Wczytanie checkpointa
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# Mapowanie etykiet
label_to_idx = checkpoint['label_to_idx']
idx_to_label = checkpoint['idx_to_label']

# Dynamiczne generowanie brand_to_id na podstawie label_to_idx
brand_to_id = {label: idx for label, idx in label_to_idx.items()}



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

    handle_fwd = last_conv.register_forward_hook(forward_hook)
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
        transforms.Resize((224, 224)),  # Rozmiar zgodny z IMAGE_SIZE w tune_model.py
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Keep a copy of the original image and size
    orig_image = image.convert('RGB')
    orig_size = orig_image.size  # (width, height)

    # Model input tensor (224x224)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Forward to get probabilities (no grad needed)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)

    confidence, predicted = torch.max(probabilities, 1)
    predicted_class = idx_to_label[predicted.item()]
    predicted_id = brand_to_id.get(predicted_class, "Nieznane ID")
    confidence_percent = confidence.item() * 100

    print(f"\nModel przewiduje: {predicted_class}")
    print(f"ID przewidywane: {predicted_id}")
    print(f"Pewność: {confidence_percent:.2f}%")

    # Compute Grad-CAM: use a tensor that requires grad
    image_tensor_for_cam = image_tensor.clone().detach().requires_grad_(True)
    # generate_gradcam will compute map and return normalized float32 HxW in 0..1 at 224x224
    gradcam_map = generate_gradcam(image_tensor_for_cam, model, predicted.item())

    # Apply a colormap to the Grad-CAM map and resize to original image size
    cmap = plt.get_cmap('jet')
    heatmap_rgb = cmap(gradcam_map)[:, :, :3]  # HxWx3 float
    heatmap_uint8 = np.uint8(heatmap_rgb * 255)
    heatmap_img = Image.fromarray(heatmap_uint8).resize(orig_size, resample=Image.BILINEAR)

    # Blend heatmap with original image (both RGB)
    blended = Image.blend(orig_image, heatmap_img.convert('RGB'), alpha=0.5)

    return {
        "brand": predicted_class,
        "confidence": confidence_percent,
        "heatmap": blended
    }