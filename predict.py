import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import io

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
def generate_gradcam(image_tensor, model, target_class):
    # Hooki do wyciągnięcia aktywacji i gradientów z ostatniej warstwy konwolucyjnej
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    # Znajdź ostatnią warstwę konwolucyjną
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    # Rejestruj hooki
    handle_fwd = last_conv.register_forward_hook(forward_hook)
    handle_bwd = last_conv.register_backward_hook(backward_hook)

    # Forward
    output = model(image_tensor)
    # Backward dla target_class
    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()

    # Wyciągnij aktywacje i gradienty
    act = activations[0]         # shape: [1, C, H, W]
    grad = gradients[0]          # shape: [1, C, H, W]
    weights = grad.mean(dim=(2, 3), keepdim=True)  # shape: [1, C, 1, 1]
    gradcam_map = (weights * act).sum(dim=1, keepdim=True)  # shape: [1, 1, H, W]
    gradcam_map = torch.relu(gradcam_map)
    gradcam_map = gradcam_map.squeeze().cpu().numpy()
    gradcam_map = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min() + 1e-8)  # Normalizacja

    # Resize do 224x224 i konwersja na heatmapę
    gradcam_map = np.uint8(255 * gradcam_map)
    gradcam_map = Image.fromarray(gradcam_map).resize((224, 224), resample=Image.BILINEAR)
    gradcam_map = np.array(gradcam_map)

    # Oryginalny obraz (po transformacji odwrotnej)
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img_np = inv_normalize(image_tensor[0].cpu()).clamp(0, 1).permute(1, 2, 0).numpy()
    img_np = np.uint8(255 * img_np)

    # Nałożenie heatmapy na obraz
    cmap = plt.get_cmap('jet')
    heatmap = cmap(gradcam_map/255.0)[:, :, :3]
    overlay = np.uint8(0.5 * img_np + 0.5 * (heatmap * 255))

    # Konwersja do PIL.Image
    overlay_img = Image.fromarray(overlay)

    # Zwolnij hooki
    handle_fwd.remove()
    handle_bwd.remove()

    return overlay_img

# === FUNKCJA DO PREDYKCJI ===
def predict_image(image):
    # Transformacje obrazu (zgodne z tune_model.py)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Rozmiar zgodny z IMAGE_SIZE w tune_model.py
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Przetworzenie obrazu
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)

    # Najbardziej prawdopodobna klasa
    confidence, predicted = torch.max(probabilities, 1)
    predicted_class = idx_to_label[predicted.item()]
    predicted_id = brand_to_id.get(predicted_class, "Nieznane ID")
    confidence_percent = confidence.item() * 100

    print(f"\nModel przewiduje: {predicted_class}")
    print(f"ID przewidywane: {predicted_id}")
    print(f"Pewność: {confidence_percent:.2f}%")

    # Grad-CAM heatmap
    heatmap_img = generate_gradcam(image_tensor, model, predicted.item())

    # Zwrócenie przewidywanej klasy, procentu pewności i heatmapy
    return {
        "brand": predicted_class,
        "confidence": confidence_percent,
        "heatmap": heatmap_img
    }