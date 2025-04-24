import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import os

# Ścieżka do zapisanego modelu
MODEL_PATH = "car_model.pth"
# Ścieżka do folderu z obrazami do testowania
IMAGE_FOLDER = "/home/tomasz/Pulpit/PROJEKT_TS/PROJEKT_TS/Data/testxd/"

# Wczytaj model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 10)  # 10 klas

# Załaduj stan modelu
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()  # Ustaw model w tryb ewaluacji

# Wczytaj etykiety
label_to_idx = checkpoint['label_to_idx']
idx_to_label = checkpoint['idx_to_label']

# Słownik mapujący marki na ID
brand_to_id = {
    "TOYOTA": 6,
    "RENAULT": 8,
    "VOLKSWAGEN": 0,
    "OPEL": 1,
    "KIA": 7,
    "FORD": 5,
    "AUDI": 2,
    "BMW": 3,
    "HYUNDAI": 4,
    "SKODA": 9
}

# Funkcja predykcji dla obrazu
def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Dodaj wymiar batcha

    image = image.to(device)
    with torch.no_grad():  # Zablokuj obliczanie gradientów
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    # Odczyt klasy na podstawie ID
    predicted_class = idx_to_label[predicted.item()]
    predicted_id = brand_to_id.get(predicted_class, "Nieznane ID")
    
    return predicted_class, predicted_id

# Sprawdź każdy folder i plik w folderze
def check_images_in_folder(folder_path):
    correct_predictions = 0
    total_predictions = 0
    
    for brand_folder in os.listdir(folder_path):
        brand_folder_path = os.path.join(folder_path, brand_folder)
        
        if os.path.isdir(brand_folder_path):  # Upewnij się, że to folder
            for filename in os.listdir(brand_folder_path):
                if filename.endswith(".jpg"):  # Tylko pliki .jpg
                    image_path = os.path.join(brand_folder_path, filename)
                    predicted_class, predicted_id = predict_image(image_path)
                    
                    # Pobierz pierwsze 4 litery z nazwy pliku jako markę
                    true_class = filename[:4]  # Pierwsze cztery litery nazwy pliku
                    
                    # Porównaj przewidywaną markę z prawdziwą
                    if predicted_class == true_class:
                        correct_predictions += 1
                    
                    total_predictions += 1
                    
                    print(f"Plik: {filename} w folderze {brand_folder}")
                    print(f"Model przewiduje: {predicted_class}")
                    print(f"ID przewidywane: {predicted_id}")
                    print(f"Prawdziwa marka: {true_class}")
                    print(f"Poprawność: {'Tak' if predicted_class == true_class else 'Nie'}")
                    print("-" * 30)
    
    # Podsumowanie dokładności
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Dokładność modelu: {accuracy * 100:.2f}%")
    
# Testowanie folderu
check_images_in_folder(IMAGE_FOLDER)
