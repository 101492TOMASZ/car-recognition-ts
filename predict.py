import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

# === KONFIGURACJA ===
MODEL_PATH = r"model\car_model2.pth"  # Ścieżka do zapisanego modelu
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



# === FUNKCJA DO PREDYKCJI ===
def predict_image(image):
    # Transformacje obrazu (zgodne z tune_model.py)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Rozmiar zgodny z IMAGE_SIZE w tune_model.py
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Przetworzenie obrazu
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)


    # Najbardziej prawdopodobna klasa
    confidence, predicted = torch.max(probabilities, 1)
    predicted_class = idx_to_label[predicted.item()]
    predicted_id = brand_to_id.get(predicted_class, "Nieznane ID")
    confidence_percent = confidence.item() * 100

    print(f"\nModel przewiduje: {predicted_class}")
    print(f"ID przewidywane: {predicted_id}")
    print(f"Pewność: {confidence_percent:.2f}%")

    # Zwrócenie przewidywanej klasy i procentu pewności
    return {
        "brand": predicted_class,
        "confidence": confidence_percent
    }