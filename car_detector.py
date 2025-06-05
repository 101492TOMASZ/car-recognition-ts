from ultralytics import YOLO
import torch

class CarDetector:
    def __init__(self):
        self.model = YOLO('yolov8s.pt')
        self.car_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck class indices

    def detect_car(self, image_path):
        """
        Checks if image contains a vehicle
        Returns: (bool, float) - (has_car, confidence)
        """
        results = self.model(image_path)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls in self.car_classes and conf > 0.3:  # confidence threshold
                    return True, conf
        
        return False, 0.0