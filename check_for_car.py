import torch
import cv2
import numpy as np
import os

# Załaduj model YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Załaduj obraz (tutaj przykładowe zdjęcie)
img = cv2.imread('OIP.jpg')

# Wykonaj detekcję obiektów
results = model(img)

# Przekształć wyniki detekcji do formatu numpy
detections = results.xyxy[0].cpu().numpy()  # XYXY format (x1, y1, x2, y2, confidence, class)

# Definiuj klasy dla samochodów (klasa 2 to "car" w YOLOv5)
car_class = 2

# Wyciągnij detekcje samochodów (tylko klasa "car")
cars = [det for det in detections if int(det[5]) == car_class]

# Jeśli nie wykryto samochodów
if len(cars) == 0:
    print("Brak samochodów na zdjęciu.")
else:
    # Wyznacz środek obrazu
    height, width, _ = img.shape
    center_x = width // 2
    center_y = height // 2

    # Wybór samochodu najbliżej środka obrazu
    closest_car = None
    min_distance = float('inf')

    for car in cars:
        # Wylicz środek wykrytego samochodu
        x_center = (car[0] + car[2]) / 2
        y_center = (car[1] + car[3]) / 2
        
        # Oblicz odległość do środka obrazu
        distance = np.sqrt((x_center - center_x) ** 2 + (y_center - center_y) ** 2)
        
        if distance < min_distance:
            closest_car = car
            min_distance = distance

    # Rysowanie prostokąta wokół najbliższego samochodu
    if closest_car is not None:
        x1, y1, x2, y2, _, _ = closest_car
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(img, "Najblizszy samochod", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Wycięcie obrazu do rozmiaru tabelki wokół samochodu
        cropped_img = img[int(y1):int(y2), int(x1):int(x2)]

        #Wyświetlenie przyciętego obrazu (tabelki wokół samochodu)
        #cv2.imshow("Wykryty samochod", cropped_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # Ścieżka zapisu obrazu
        output_path = '/home/tomasz/Pulpit/PROJEKT_TS/PROJEKT_TS/cropped_car.jpg'
        # Zapisz obraz, sprawdź, czy zapis był pomyślny
        success = cv2.imwrite(output_path, cropped_img)

        if success:
            print(f"Obraz zapisano pomyślnie: {output_path}")
        else:
            print(f"Nie udało się zapisać obrazu: {output_path}")
    else:
        print("Nie znaleziono samochodu w pobliżu środka.")
