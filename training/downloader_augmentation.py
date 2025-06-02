from icrawler.builtin import BingImageCrawler
import os
from PIL import Image, ImageFilter
import numpy as np
import random
from torchvision.transforms import functional as F

# Lista marek do pobrania
brands = [
    "BMW", "Audi", "Toyota", "Volkswagen", "Mercedes",
    "Ford", "Opel", "Hyundai", "Kia", "Skoda"
]

# Lista dodatkowych tagów do wyszukiwania
tags = ["car", "vehicle", "automobile", "sedan", "SUV",
        "hatchback", "coupe", "convertible", "pickup", "truck", "2010s", "2020s", "2000s"]

# Liczba obrazów do pobrania na każdą markę i tag
images_per_tag = 100  # Liczba obrazów dla każdego tagu

# Folder do zapisu obrazów
output_dir = "dataset"
temp_dir = "temp"  # Folder tymczasowy na oryginalne obrazy

# Funkcja augmentacji z 80% szansą na zmiany i zmianą rozmiaru do 224x224
def augment_image(img):
    # Zmiana rozmiaru do 224x224
    img = img.resize((224, 224))  # Przeskalowanie obrazu

    # Zmiana jasności
    if random.random() < 0.5:  # 50% szansy
        img = F.adjust_brightness(img, random.uniform(0.4, 1.7))  # Jasność 40%-170%

    # Zmiana kontrastu
    if random.random() < 0.5:  # 50% szansy
        img = F.adjust_contrast(img, random.uniform(0.4, 1.7))  # Kontrast 40%-170%

    # Dodanie szumu
    if random.random() < 0.2:  # 20% szansy
        img_array = np.array(img)  # Konwersja obrazu do macierzy numpy
        noise = np.random.normal(0, 25, img_array.shape).astype(np.int16)  # Szum o tym samym kształcie co obraz
        img_array = img_array + noise  # Dodanie szumu
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)  # Przycięcie wartości do zakresu [0, 255]
        img = Image.fromarray(img_array)  # Konwersja z powrotem do obrazu

    # Obrót
    if random.random() < 0.1:  # 10% szansy
        angle = random.uniform(-25, 25)
        if isinstance(img, Image.Image):
            img = img.rotate(angle)
        else:
            img = F.rotate(img, angle)

    # Odbicie poziome
    if random.random() < 0.5:  # 50% szansy
        from PIL import ImageOps
        if not isinstance(img, Image.Image):
            img = F.to_pil_image(img)
        img = ImageOps.mirror(img)

    # Upewnij się, że wynik to PIL Image
    if not isinstance(img, Image.Image):
        img = F.to_pil_image(img)
    return img
    if random.random() < 0.1:  # 50% szansy
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 3)))  # Większe rozmycie

    return img

# Pobieranie obrazów z Bing
for brand in brands:
    brand_dir = os.path.join(output_dir, brand.upper())  # Folder docelowy dla augmentowanych obrazów
    os.makedirs(brand_dir, exist_ok=True)  # Tworzenie folderu docelowego, jeśli nie istnieje

    temp_brand_dir = os.path.join(temp_dir, brand.upper())  # Folder tymczasowy dla oryginalnych obrazów
    os.makedirs(temp_brand_dir, exist_ok=True)  # Tworzenie folderu tymczasowego, jeśli nie istnieje

    # Pobranie obrazów dla każdej kombinacji marki i tagu
    for tag in tags:
        keyword = f"{brand} {tag}"
        print(f"Rozpoczęto pobieranie obrazów dla: {keyword}")

        try:
            # Inicjalizacja crawlera dla Bing
            crawler = BingImageCrawler(storage={'root_dir': temp_brand_dir})
            crawler.crawl(keyword=keyword, max_num=images_per_tag)

            # Przetwarzanie obrazów w locie
            for idx, img_name in enumerate(os.listdir(temp_brand_dir)):
                try:
                    img_path = os.path.join(temp_brand_dir, img_name)
                    img = Image.open(img_path).convert('RGB')  # Wczytaj obraz z folderu tymczasowego
                    augmented_img = augment_image(img)  # Zastosuj augmentację

                    # Zapisz obraz z unikalną nazwą w folderze docelowym
                    new_img_name = f"aug_{brand}_{tag}_{idx}.jpg"
                    augmented_img.save(os.path.join(brand_dir, new_img_name))

                    # Zmień nazwę oryginalnego pliku w folderze tymczasowym, aby uniknąć konfliktów
                    temp_img_name = f"temp_{brand}_{tag}_{idx}.jpg"
                    os.rename(img_path, os.path.join(temp_brand_dir, temp_img_name))
                except Exception as e:
                    print(f"Błąd przetwarzania obrazu {img_name}: {e}")
        except Exception as e:
            print(f"Błąd podczas pobierania obrazów dla {keyword}: {e}")
            continue

    print(f"Pobrano i przetworzono obrazy dla marki: {brand}")