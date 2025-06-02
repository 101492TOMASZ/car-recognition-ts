from icrawler.builtin import BingImageCrawler
import os

# Lista marek do pobrania
brands = [
    "BMW", "Audi", "Toyota", "Volkswagen", "Mercedes",
    "Ford", "Opel", "Hyundai", "Kia", "Skoda"
]

# Lista dodatkowych tagów do wyszukiwania
tags = ["car", "vehicle", "automobile", "sedan", "SUV",
        "hatchback", "coupe", "convertible", "pickup", "truck", "2010s", "2020s", "2000s"]

# Liczba obrazów do pobrania na każdą markę i tag
images_per_tag = 200  # Liczba obrazów dla każdego tagu

# Folder do zapisu obrazów
output_dir = "dataset"

# Pobieranie obrazów z Bing
for brand in brands:
    brand_dir = os.path.join(output_dir, brand.upper())  # Tworzy folder dla każdej marki
    os.makedirs(brand_dir, exist_ok=True)  # Jeśli folder nie istnieje, zostanie utworzony

    # Liczba istniejących plików w folderze
    existing_files = len(os.listdir(brand_dir))
    
    # Pobranie obrazów dla każdej kombinacji marki i tagu
    for tag in tags:
        keyword = f"{brand} {tag}"
        print(f"Pobieranie obrazów dla: {keyword}")

        # Inicjalizacja crawlera dla Bing
        crawler = BingImageCrawler(storage={'root_dir': brand_dir})
        
        # Pobranie obrazów dla danego tagu z przesunięciem indeksów
        crawler.crawl(keyword=keyword, max_num=images_per_tag, file_idx_offset=existing_files)
        existing_files += images_per_tag  # Aktualizacja liczby istniejących plików
    
    print(f"Pobrano obrazy dla marki: {brand}")
