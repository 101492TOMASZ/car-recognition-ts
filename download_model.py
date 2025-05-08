import gdown
import os

# Ustawienie ścieżki docelowej
url = 'https://drive.google.com/uc?id=155xTY8ma9HE2b78Ltlmh1Kb0vcECpkkU'
output_folder = 'model'
output_file = 'car_model2.pth'
output_path = os.path.join(output_folder, output_file)

# Tworzenie folderu, jeśli nie istnieje
os.makedirs(output_folder, exist_ok=True)

# Sprawdzenie, czy plik już istnieje
if os.path.exists(output_path):
    print(f"Plik '{output_file}' już istnieje w folderze '{output_folder}'.")
else:
    # Pobranie pliku do folderu 'model'
    print(f"Pobieranie pliku '{output_file}'...")
    gdown.download(url, output_path, quiet=False)
    print(f"Pobrano plik do '{output_path}'.")
