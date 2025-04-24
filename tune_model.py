import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# === KONFIGURACJA ===
DATASET_DIR = r"/home/tomasz/Pulpit/PROJEKT_TS/PROJEKT_TS/dataset"
BATCH_SIZE = 32
EPOCHS = 10  # Liczba epok do kontynuacji
LR = 0.0005
IMAGE_SIZE = (224, 224)
SAVE_PATH = "car_model2.pth"
VERBOSE = False  # ustaw na True, aby drukować każdą próbkę

# === DANE ===
class CarDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.label_to_idx = {}
        self.idx_to_label = {}
        idx = 0

        # Iteracja po folderach w katalogu głównym
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            # Marka to nazwa folderu
            brand = folder.upper()
            if brand not in self.label_to_idx:
                self.label_to_idx[brand] = idx
                self.idx_to_label[idx] = brand
                idx += 1

            label_idx = self.label_to_idx[brand]

            # Rekurencyjne przeszukiwanie folderów
            for root, _, files in os.walk(folder_path):
                for fname in files:
                    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
                        file_path = os.path.join(root, fname)
                        try:
                            with Image.open(file_path) as img:
                                self.samples.append((file_path, label_idx))
                                if VERBOSE:
                                    print(f"Plik: {file_path} ➤ Marka: {brand}")
                        except Exception as e:
                            print(f"Błąd wczytywania pliku {file_path}: {e}")

        print(f"Liczba załadowanych próbek: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# === TRANSFORMACJE Z AUGMENTACJĄ ===
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === ŁADOWANIE DANYCH I MODELU ===
dataset = CarDataset(DATASET_DIR, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
num_classes = len(dataset.label_to_idx)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Używane urządzenie:", device)

model = models.resnet50(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# === Wczytywanie zapisanego modelu ===
if os.path.exists(SAVE_PATH):
    print(f"Wczytywanie modelu z {SAVE_PATH}...")
    checkpoint = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", optimizer.state_dict()))
    dataset.label_to_idx = checkpoint["label_to_idx"]
    dataset.idx_to_label = checkpoint["idx_to_label"]
    print("Model i optymalizator wczytane pomyślnie.")
else:
    print("Brak zapisanego modelu. Rozpoczynam trening od nowa.")

# === FINE-TUNING ===
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"Epoka {epoch}/{EPOCHS}")
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Obliczanie predykcji
        _, predicted = torch.max(outputs.data, 1)

        # Wyświetlanie informacji o przewidywaniach i prawdziwych etykietach
        for i in range(len(labels)):
            predicted_label = dataset.idx_to_label[predicted[i].item()]
            true_label = dataset.idx_to_label[labels[i].item()]
            print(f"PRZEWIDYWANY MODEL: {predicted_label} | PRAWDZIWY: {true_label}")

        # Aktualizacja dokładności
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        total_loss += loss.item()

        # Aktualizacja paska postępu
        progress_bar.set_postfix({
            "Strata": f"{total_loss / (batch_idx + 1):.4f}",
            "Dokładność": f"{100 * correct / total:.2f}%"
        })

    # Wyświetlanie podsumowania epoki
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    print(f"\n=== Epoka {epoch}/{EPOCHS} ===")
    print(f"Strata: {epoch_loss:.4f}")
    print(f"Dokładność: {epoch_acc:.2f}%")

    # Zapis modelu po każdej epoce
    epoch_save_path = f"car_model_epoch_{epoch}.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "label_to_idx": dataset.label_to_idx,
        "idx_to_label": dataset.idx_to_label
    }, epoch_save_path)
    print(f"Model zapisany po epoce {epoch} jako {epoch_save_path}")

# === ZAPISYWANIE ===
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "label_to_idx": dataset.label_to_idx,
    "idx_to_label": dataset.idx_to_label
}, SAVE_PATH)

print(f"Model zapisany jako {SAVE_PATH}")